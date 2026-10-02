from typing import List, Optional, Tuple, Union

import torch
from torch import nn


class QwenImageScoreWrapper(nn.Module):
    """
    Frozen Qwen-Image score model for latent-level image DMD.

    Inputs use Wan/Qwen normalized latent space with shape [B, F, C, H, W].
    Qwen-Image packs each single-frame latent into 2x2 spatial patches before
    passing it to the transformer.
    """

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        max_sequence_length: int = 512,
    ):
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage import (
            QwenImageTransformer2DModel,
        )
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

        self.model_path = model_path
        self.dtype = dtype
        self.max_sequence_length = max_sequence_length
        self.prompt_template_encode = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = 34
        self.tokenizer_max_length = 1024

        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            f"{model_path}/tokenizer"
        )
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            f"{model_path}/text_encoder",
            torch_dtype=dtype,
        ).eval().requires_grad_(False)
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            f"{model_path}/transformer",
            torch_dtype=dtype,
        ).eval().requires_grad_(False)

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, H/2 * W/2, C * 4]
        batch_size, channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Qwen latent height/width must be even, got {height}x{width}")
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # [B, H/2 * W/2, C * 4] -> [B, C, H, W]
        batch_size, _, channels = latents.shape
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels // 4, height, width)

    @staticmethod
    def _extract_masked_hidden(hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    @torch.no_grad()
    def encode_prompt(
        self,
        text_prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        txt = [self.prompt_template_encode.format(prompt) for prompt in text_prompts]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, txt_tokens.attention_mask
        )
        split_hidden_states = [
            hidden[self.prompt_template_encode_start_idx:]
            for hidden in split_hidden_states
        ]
        attn_mask_list = [
            torch.ones(hidden.size(0), dtype=torch.long, device=hidden.device)
            for hidden in split_hidden_states
        ]
        max_seq_len = min(
            max(hidden.size(0) for hidden in split_hidden_states),
            self.max_sequence_length,
        )
        prompt_embeds = torch.stack(
            [
                torch.cat(
                    [
                        hidden[:max_seq_len],
                        hidden.new_zeros(max_seq_len - min(hidden.size(0), max_seq_len), hidden.size(1)),
                    ]
                )
                for hidden in split_hidden_states
            ]
        )
        prompt_mask = torch.stack(
            [
                torch.cat(
                    [
                        mask[:max_seq_len],
                        mask.new_zeros(max_seq_len - min(mask.size(0), max_seq_len)),
                    ]
                )
                for mask in attn_mask_list
            ]
        )
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if prompt_mask.all():
            prompt_mask = None
        return prompt_embeds, prompt_mask

    def _predict_flow(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, _, height, width = noisy_latent.shape
        packed_latent = self._pack_latents(noisy_latent)
        img_shapes = [[(1, height // 2, width // 2)]] * batch_size
        with self.transformer.cache_context("cond"):
            packed_flow = self.transformer(
                hidden_states=packed_latent,
                timestep=timestep.to(dtype=packed_latent.dtype) / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                attention_kwargs={},
                return_dict=False,
            )[0]
        return self._unpack_latents(packed_flow, height, width)

    @torch.no_grad()
    def forward(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        text_prompts: List[str],
        negative_prompt: str = " ",
        guidance_scale: float = 1.0,
        return_conditional_prediction: bool = False,
        return_conditional_and_unconditional_predictions: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Return x0 prediction for [B, F, C, H, W] latent inputs.

        When ``return_conditional_prediction`` is true, return
        ``(cfg_x0, conditional_x0)``.  Decoupled DMD needs both predictions
        at the CA timestep, while the default keeps the original tensor-only
        API and behavior.

        ``return_conditional_and_unconditional_predictions`` returns the raw
        ``(conditional_x0, unconditional_x0)`` pair before CFG normalization,
        as required by the CA term in Decoupled DMD.
        """
        batch_size, num_frames, channels, height, width = noisy_latent.shape
        flat_noisy = noisy_latent.reshape(batch_size * num_frames, channels, height, width)
        flat_timestep = timestep.reshape(batch_size * num_frames)
        flat_prompts = [
            prompt for prompt in text_prompts for _ in range(num_frames)
        ]

        dtype = next(self.transformer.parameters()).dtype
        device = flat_noisy.device
        prompt_embeds, prompt_mask = self.encode_prompt(flat_prompts, device, dtype)
        pred_flow = self._predict_flow(
            flat_noisy.to(dtype=dtype),
            flat_timestep,
            prompt_embeds,
            prompt_mask,
        )
        conditional_flow = pred_flow

        negative_flow = None
        if guidance_scale > 1.0 or return_conditional_and_unconditional_predictions:
            negative_prompts = [negative_prompt] * len(flat_prompts)
            neg_embeds, neg_mask = self.encode_prompt(negative_prompts, device, dtype)
            negative_flow = self._predict_flow(
                flat_noisy.to(dtype=dtype),
                flat_timestep,
                neg_embeds,
                neg_mask,
            )
        if guidance_scale > 1.0:
            cfg_flow = negative_flow + guidance_scale * (
                pred_flow - negative_flow
            )
            cond_norm = torch.norm(pred_flow.flatten(1), dim=1, keepdim=True).clamp_min(1e-6)
            cfg_norm = torch.norm(cfg_flow.flatten(1), dim=1, keepdim=True).clamp_min(1e-6)
            pred_flow = cfg_flow * (cond_norm / cfg_norm).view(-1, 1, 1, 1)

        sigma = (flat_timestep / 1000).to(device=device, dtype=torch.float32)
        sigma = sigma.reshape(-1, 1, 1, 1).to(dtype=pred_flow.dtype)
        pred_x0 = flat_noisy.to(dtype=pred_flow.dtype) - sigma * pred_flow
        pred_x0 = pred_x0.reshape(
            batch_size, num_frames, channels, height, width
        ).to(noisy_latent.dtype)
        if not (
            return_conditional_prediction
            or return_conditional_and_unconditional_predictions
        ):
            return pred_x0

        conditional_x0 = flat_noisy.to(dtype=conditional_flow.dtype) - (
            sigma.to(dtype=conditional_flow.dtype) * conditional_flow
        )
        conditional_x0 = conditional_x0.reshape(
            batch_size, num_frames, channels, height, width
        ).to(noisy_latent.dtype)
        if return_conditional_and_unconditional_predictions:
            unconditional_x0 = flat_noisy.to(dtype=negative_flow.dtype) - (
                sigma.to(dtype=negative_flow.dtype) * negative_flow
            )
            unconditional_x0 = unconditional_x0.reshape(
                batch_size, num_frames, channels, height, width
            ).to(noisy_latent.dtype)
            return conditional_x0, unconditional_x0
        return pred_x0, conditional_x0
