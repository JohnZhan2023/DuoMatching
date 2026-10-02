from pipeline import SelfForcingTrainingPipeline
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torch
import torch.distributed as dist

from model.base import SelfForcingModel
from utils.qwen_timestep_scheduler import (
    load_qwen_scheduler_config,
    map_qwen_raw_timestep,
)


class DMD(SelfForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: SelfForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

        self.qwen_image_teacher_enabled = False
        self.qwen_image_score = None
        self.qwen_image_fake_score = None
        self.qwen_image_fake_critic_enabled = False
        self.qwen_image_fake_critic_loss_weight = 0.0
        self.qwen_frame_latent_adapter = None
        qwen_cfg = getattr(args, "qwen_image_teacher", None)
        if qwen_cfg is not None and getattr(qwen_cfg, "enabled", False):
            from utils.qwen_image_wrapper import QwenImageScoreWrapper

            self.qwen_image_teacher_enabled = True
            self.qwen_image_frames_per_video = int(getattr(qwen_cfg, "frames_per_video", 1))
            include_first_frame = getattr(qwen_cfg, "include_first_frame", False)
            self.qwen_image_random_include_first_frame = False
            if isinstance(include_first_frame, str):
                include_first_frame_mode = (
                    include_first_frame.lower().replace("-", "_")
                )
                if include_first_frame_mode in (
                    "random",
                    "random_candidate",
                    "candidate",
                    "eligible",
                ):
                    self.qwen_image_include_first_frame = False
                    self.qwen_image_random_include_first_frame = True
                elif include_first_frame_mode in ("always", "anchor", "true"):
                    self.qwen_image_include_first_frame = True
                elif include_first_frame_mode in ("never", "false"):
                    self.qwen_image_include_first_frame = False
                else:
                    raise ValueError(
                        "qwen_image_teacher.include_first_frame must be a "
                        "boolean or one of 'random', 'always', or 'never', got "
                        f"{include_first_frame!r}"
                    )
            else:
                self.qwen_image_include_first_frame = bool(include_first_frame)
            self.qwen_image_frame_sampling = getattr(
                qwen_cfg, "frame_sampling", "random"
            )
            self.qwen_image_frame_difference_metric = getattr(
                qwen_cfg, "frame_difference_metric", "mean_abs"
            )
            self.qwen_image_transition_peak_min_distance = int(
                getattr(qwen_cfg, "transition_peak_min_distance", 3)
            )
            self.qwen_image_transition_boundary_margin = int(
                getattr(qwen_cfg, "transition_boundary_margin", 1)
            )
            self.qwen_image_log_frame_sampling = bool(
                getattr(qwen_cfg, "log_frame_sampling", False)
            )
            self.qwen_image_stratified_include_first_frame = bool(
                getattr(qwen_cfg, "stratified_include_first_frame", False)
            )
            self.qwen_image_loss_weight = float(getattr(qwen_cfg, "loss_weight", 0.1))
            self.qwen_image_warmup_steps = int(getattr(qwen_cfg, "warmup_steps", 0))
            self.qwen_image_fake_critic_enabled = bool(
                getattr(qwen_cfg, "train_fake_critic", False)
            )
            self.qwen_image_fake_critic_loss_weight = float(
                getattr(qwen_cfg, "fake_critic_loss_weight", 1.0)
            )
            if self.qwen_image_fake_critic_loss_weight < 0:
                raise ValueError(
                    "qwen_image_teacher.fake_critic_loss_weight must be >= 0"
                )
            self.qwen_image_guidance_scale = float(
                getattr(qwen_cfg, "guidance_scale", self.real_guidance_scale)
            )
            self.decoupled_dmd_for_qwen = bool(
                getattr(qwen_cfg, "decoupled_dmd_for_qwen", False)
            )
            self.qwen_image_timestep_sampling = getattr(
                qwen_cfg, "timestep_sampling", "random"
            )
            self.qwen_image_timestep_schedule = str(
                getattr(qwen_cfg, "timestep_schedule", "qwen_dynamic")
            ).lower().replace("-", "_")
            self.qwen_image_scheduler_reference_steps = int(
                getattr(qwen_cfg, "scheduler_reference_steps", 50)
            )
            self.qwen_image_negative_prompt = getattr(
                qwen_cfg, "negative_prompt", " "
            )
            self.qwen_image_scheduler_config = None
            if self.qwen_image_timestep_schedule in (
                "qwen_dynamic",
                "qwen_native",
            ):
                self.qwen_image_scheduler_config = load_qwen_scheduler_config(
                    getattr(qwen_cfg, "model_path")
                )
            elif self.qwen_image_timestep_schedule not in (
                "wan_shift",
                "legacy_wan_shift",
            ):
                raise ValueError(
                    "qwen_image_teacher.timestep_schedule must be "
                    "'qwen_dynamic' or 'wan_shift', got "
                    f"{self.qwen_image_timestep_schedule!r}"
                )
            if (
                self.decoupled_dmd_for_qwen
                and self.qwen_image_timestep_schedule
                in ("qwen_dynamic", "qwen_native")
            ):
                raise ValueError(
                    "decoupled_dmd_for_qwen currently uses Wan-coordinate "
                    "timestep bounds and is not yet compatible with "
                    "timestep_schedule=qwen_dynamic"
                )
            self.qwen_image_latent_adapter = str(
                getattr(qwen_cfg, "latent_adapter", "direct")
            ).lower().replace("-", "_")
            if self.qwen_image_latent_adapter not in (
                "direct",
                "frame_latent_adapter",
            ):
                raise ValueError(
                    "qwen_image_teacher.latent_adapter must be 'direct' or "
                    f"'frame_latent_adapter', got {self.qwen_image_latent_adapter!r}"
                )
            if self.qwen_image_latent_adapter == "frame_latent_adapter":
                from utils.frame_latent_adapter import FrameLatentAdapter

                adapter_path = getattr(qwen_cfg, "adapter_model_path", None)
                if not adapter_path:
                    raise ValueError(
                        "qwen_image_teacher.adapter_model_path is required when "
                        "latent_adapter=frame_latent_adapter"
                    )
                self.qwen_frame_latent_adapter = (
                    FrameLatentAdapter.from_pretrained(adapter_path)
                    .to(device=device, dtype=self.dtype)
                    .eval()
                    .requires_grad_(False)
                )
            self.qwen_image_score = QwenImageScoreWrapper(
                model_path=getattr(qwen_cfg, "model_path"),
                dtype=self.dtype,
                max_sequence_length=int(getattr(qwen_cfg, "max_sequence_length", 512)),
            )
            if self.qwen_image_fake_critic_enabled:
                from utils.wan_wrapper import WanDiffusionWrapper

                self.qwen_image_fake_score = WanDiffusionWrapper(
                    model_name=getattr(
                        qwen_cfg, "fake_model_name", self.fake_model_name
                    ),
                    is_causal=False,
                )
                self.qwen_image_fake_score.model.requires_grad_(True)
                if args.gradient_checkpointing:
                    self.qwen_image_fake_score.enable_gradient_checkpointing()


    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach()
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict, torch.Tensor]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            # Useless if we set always 21 latent frames
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict, grad

    @staticmethod
    def _dmd_loss_from_grad(
        original_latent: torch.Tensor,
        grad: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target = (original_latent.double() - grad.double()).detach()
        if gradient_mask is not None:
            return 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                target[gradient_mask],
                reduction="mean",
            )
        return 0.5 * F.mse_loss(
            original_latent.double(),
            target,
            reduction="mean",
        )

    @staticmethod
    def _qwen_video_wan_gradient_logs(
        qwen_grad: torch.Tensor,
        video_wan_grad: torch.Tensor,
        frame_indices: torch.Tensor,
        qwen_loss_weight: float,
        qwen_mask: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compare sparse Qwen gradients with the full-video Wan DMD gradient."""
        qwen_grad = qwen_grad.float()
        video_wan_grad = video_wan_grad.float()
        if qwen_mask is not None:
            qwen_grad = qwen_grad * qwen_mask.to(dtype=qwen_grad.dtype)
        if video_mask is not None:
            video_wan_grad = video_wan_grad * video_mask.to(
                dtype=video_wan_grad.dtype
            )

        gather_index = frame_indices[:, :, None, None, None].expand(
            -1, -1, *video_wan_grad.shape[2:]
        )
        video_support = torch.gather(video_wan_grad, dim=1, index=gather_index)

        qwen_flat = qwen_grad.flatten(2)
        support_flat = video_support.flatten(2)
        qwen_support_l2 = torch.linalg.vector_norm(qwen_flat, dim=2)
        wan_support_l2 = torch.linalg.vector_norm(support_flat, dim=2)
        support_dot = (qwen_flat * support_flat).sum(dim=2)
        eps = 1e-8
        support_cosine = support_dot / (
            qwen_support_l2 * wan_support_l2
        ).clamp_min(eps)

        qwen_full = torch.zeros_like(video_wan_grad)
        qwen_full.scatter_add_(dim=1, index=gather_index, src=qwen_grad)
        qwen_global_flat = qwen_full.flatten(1)
        wan_global_flat = video_wan_grad.flatten(1)
        qwen_global_l2 = torch.linalg.vector_norm(qwen_global_flat, dim=1)
        wan_global_l2 = torch.linalg.vector_norm(wan_global_flat, dim=1)
        global_cosine = (qwen_global_flat * wan_global_flat).sum(dim=1) / (
            qwen_global_l2 * wan_global_l2
        ).clamp_min(eps)

        # Both proxy losses use mean reduction. A K-frame Qwen loss therefore
        # scales each supported element F/K times more strongly than the
        # F-frame video loss before applying qwen_loss_weight.
        if video_mask is None:
            video_element_count = torch.full_like(
                wan_global_l2, video_wan_grad[0].numel()
            )
        else:
            video_element_count = video_mask.flatten(1).sum(dim=1).float()
        if qwen_mask is None:
            qwen_element_count = torch.full_like(
                qwen_global_l2, qwen_grad[0].numel()
            )
        else:
            qwen_element_count = qwen_mask.flatten(1).sum(dim=1).float()
        reduction_scale = video_element_count / qwen_element_count.clamp_min(1.0)
        weighted_global_norm_ratio = (
            float(qwen_loss_weight)
            * reduction_scale
            * qwen_global_l2
            / wan_global_l2.clamp_min(eps)
        )

        return {
            "qwen_image_qwen_video_wan_support_cosine": (
                support_cosine.mean().detach()
            ),
            "qwen_image_qwen_video_wan_global_cosine": global_cosine.mean().detach(),
            "qwen_image_qwen_video_wan_negative_fraction": (
                (support_cosine < 0).float().mean().detach()
            ),
            "qwen_image_qwen_to_video_wan_weighted_norm_ratio": (
                weighted_global_norm_ratio.mean().detach()
            ),
        }

    @staticmethod
    def _repeat_conditioning_for_independent_frames(
        conditioning: dict,
        batch_size: int,
        frames_per_video: int,
    ) -> dict:
        """Repeat batch-aligned conditioning for a flattened B*K frame batch."""
        repeated = {}
        for key, value in conditioning.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                repeated[key] = value.repeat_interleave(frames_per_video, dim=0)
            else:
                repeated[key] = value
        return repeated

    def _compute_independent_frame_fake_predictions(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        score_model=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a fake critic on B*K independent one-frame samples."""
        batch_size, num_frame = noisy_latent.shape[:2]
        frame_shape = noisy_latent.shape[2:]
        flat_noisy_latent = noisy_latent.flatten(0, 1).unsqueeze(1)
        flat_timestep = timestep.flatten(0, 1).unsqueeze(1)
        flat_conditional_dict = self._repeat_conditioning_for_independent_frames(
            conditional_dict, batch_size, num_frame
        )
        if score_model is None:
            score_model = self.fake_score
        flat_flow_pred, flat_pred_fake_cond = score_model(
            noisy_image_or_video=flat_noisy_latent,
            conditional_dict=flat_conditional_dict,
            timestep=flat_timestep,
        )
        flow_pred = flat_flow_pred[:, 0].reshape(
            batch_size, num_frame, *frame_shape
        )
        x0_pred = flat_pred_fake_cond[:, 0].reshape(
            batch_size, num_frame, *frame_shape
        )
        return flow_pred, x0_pred

    def _compute_qwen_image_fake_score(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
    ) -> torch.Tensor:
        """Predict the adapter-space fake x0 with its dedicated critic when set."""
        score_model = (
            self.qwen_image_fake_score
            if self.qwen_image_fake_critic_enabled
            else self.fake_score
        )
        _, x0_pred = self._compute_independent_frame_fake_predictions(
            noisy_latent=noisy_latent,
            timestep=timestep,
            conditional_dict=conditional_dict,
            score_model=score_model,
        )
        return x0_pred

    @staticmethod
    def _qwen_per_frame_normalizer(
        original_latent: torch.Tensor,
        teacher_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize every sampled frame independently over C, H, and W."""
        return torch.abs(original_latent - teacher_prediction).mean(
            dim=[2, 3, 4], keepdim=True
        ).clamp_min(1e-6)

    def _compute_qwen_image_frame_difference(
        self,
        image_or_video: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        metric = str(self.qwen_image_frame_difference_metric).lower().replace("-", "_")
        if image_or_video.shape[1] <= 1:
            return torch.zeros(
                image_or_video.shape[0],
                candidates.numel(),
                device=image_or_video.device,
                dtype=torch.float32,
            )

        prev_indices = (candidates - 1).clamp_min(0)
        current = image_or_video.index_select(1, candidates).float()
        previous = image_or_video.index_select(1, prev_indices).float()
        delta = current - previous

        if metric in ("mean_abs", "l1", "mae"):
            score = delta.abs().mean(dim=[2, 3, 4])
        elif metric in ("rms", "l2", "rmse"):
            score = delta.square().mean(dim=[2, 3, 4]).sqrt()
        elif metric in ("cosine", "cosine_distance"):
            current_flat = current.flatten(2)
            previous_flat = previous.flatten(2)
            score = 1.0 - F.cosine_similarity(
                current_flat, previous_flat, dim=2, eps=1e-6
            )
        else:
            raise ValueError(
                "qwen_image_teacher.frame_difference_metric must be one of "
                "'mean_abs', 'rms', or 'cosine', got "
                f"{self.qwen_image_frame_difference_metric!r}"
            )

        if candidates[0].item() == 0:
            score[:, 0] = -torch.inf
        return score

    def _select_qwen_image_difference_peaks(
        self,
        scores: torch.Tensor,
        candidates: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        if num_samples >= candidates.numel():
            return candidates[None, :].expand(scores.shape[0], -1)

        left = torch.cat([scores[:, :1] - 1.0, scores[:, :-1]], dim=1)
        right = torch.cat([scores[:, 1:], scores[:, -1:] - 1.0], dim=1)
        peak_mask = (scores >= left) & (scores >= right) & torch.isfinite(scores)

        frame_indices = []
        for batch_idx in range(scores.shape[0]):
            batch_scores = scores[batch_idx]
            peak_positions = torch.nonzero(peak_mask[batch_idx], as_tuple=False).flatten()
            if peak_positions.numel() > 0:
                peak_scores = batch_scores.index_select(0, peak_positions)
                peak_order = torch.argsort(peak_scores, descending=True)
                selected_positions = peak_positions.index_select(
                    0, peak_order[:num_samples]
                )
            else:
                selected_positions = peak_positions

            if selected_positions.numel() < num_samples:
                remaining_mask = torch.ones(
                    candidates.numel(), device=scores.device, dtype=torch.bool
                )
                if selected_positions.numel() > 0:
                    remaining_mask[selected_positions] = False
                remaining_scores = batch_scores.masked_fill(~remaining_mask, -torch.inf)
                fill_count = num_samples - selected_positions.numel()
                fill_positions = torch.topk(remaining_scores, k=fill_count).indices
                selected_positions = torch.cat([selected_positions, fill_positions], dim=0)

            selected_frames = candidates.index_select(0, selected_positions[:num_samples])
            frame_indices.append(torch.sort(selected_frames).values)

        return torch.stack(frame_indices, dim=0)

    def _select_qwen_image_transition_segments(
        self,
        scores: torch.Tensor,
        candidates: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample one random stable-state frame per transition-defined segment.

        K sampled frames use K-1 peak boundaries. A peak at p denotes the
        change p-1 -> p; it partitions the sequence into a pre- and post-peak
        state.  Frames are random within each state segment rather than fixed
        at the boundary, so repeated generator updates cover its content while
        avoiding the potentially blurred transition instant.
        """
        if num_samples <= 1:
            return self._sample_qwen_image_frames_random_from_candidates(
                candidates, scores.shape[0], num_samples
            )

        # Triangular smoothing removes isolated latent-difference spikes.
        smoothed = F.avg_pool1d(
            scores.unsqueeze(1), kernel_size=3, stride=1, padding=1,
            count_include_pad=False,
        ).squeeze(1)
        required_peaks = num_samples - 1
        min_distance = max(1, self.qwen_image_transition_peak_min_distance)
        margin = max(0, self.qwen_image_transition_boundary_margin)
        sampled_all = []
        probe_rows = []
        probe_values = {
            "candidate_raw": [],
            "candidate_smoothed": [],
            "selected_raw": [],
            "selected_smoothed": [],
            "selected_prominence": [],
            "selected_positive_prominence": [],
            "selected_fallback": [],
            "selected_relaxed_nms": [],
            "selected_unsafe_edge": [],
            "selected_first_boundary": [],
            "selected_last_boundary": [],
            "local_max_count": [],
            "peak_frame": [],
            "peak_separation": [],
            "segment_length": [],
            "guarded_segment_length": [],
            "guard_fallback": [],
            "sample_frame": [],
            "sample_first_candidate": [],
            "sample_last_candidate": [],
            "sample_boundary_distance": [],
        }

        for batch_idx in range(scores.shape[0]):
            raw_row = scores[batch_idx]
            row = smoothed[batch_idx]
            # A boundary at position 0 would create an empty pre-transition
            # segment.  Only positions 1..N-1 can split the candidate frames
            # into two non-empty sets.
            valid_peak_positions = torch.arange(
                1, row.numel(), device=row.device
            )
            left = torch.cat([row[:1] - 1.0, row[:-1]], dim=0)
            right = torch.cat([row[1:], row[-1:] - 1.0], dim=0)
            local_max_mask = (
                (row >= left) & (row >= right) & torch.isfinite(row)
            )
            local_max = torch.nonzero(local_max_mask, as_tuple=False).flatten()
            local_max = local_max[local_max > 0]
            probe_values["candidate_raw"].extend(
                raw_row[valid_peak_positions].float()
            )
            probe_values["candidate_smoothed"].extend(
                row[valid_peak_positions].float()
            )
            probe_values["local_max_count"].append(float(local_max.numel()))

            # Rank peaks by local prominence, not merely raw displacement.
            # This favors a boundary that stands out from its surrounding
            # motion over a uniformly fast but non-transition segment.
            radius = min(3, max(1, candidates.numel() // 4))
            prominence = torch.empty_like(row)
            for pos in range(row.numel()):
                lo = max(0, pos - radius)
                hi = min(row.numel(), pos + radius + 1)
                neighbours = torch.cat([row[lo:pos], row[pos + 1:hi]])
                baseline = neighbours.mean() if neighbours.numel() else row[pos]
                prominence[pos] = row[pos] - baseline
            # Raw score breaks ties between equally prominent candidates.
            rank_score = prominence + 1e-3 * row
            order = local_max[torch.argsort(rank_score[local_max], descending=True)]

            peak_positions = []
            peak_sources = {}
            for pos in order.tolist():
                if all(abs(pos - existing) >= min_distance for existing in peak_positions):
                    peak_positions.append(pos)
                    peak_sources[pos] = "local_max"
                    if len(peak_positions) == required_peaks:
                        break
            if len(peak_positions) < required_peaks:
                for pos in valid_peak_positions[
                    torch.argsort(rank_score[valid_peak_positions], descending=True)
                ].tolist():
                    if pos not in peak_positions and all(
                        abs(pos - existing) >= min_distance
                        for existing in peak_positions
                    ):
                        peak_positions.append(pos)
                        peak_sources[pos] = "fallback_nms"
                        if len(peak_positions) == required_peaks:
                            break
            if len(peak_positions) < required_peaks:
                for pos in valid_peak_positions[
                    torch.argsort(rank_score[valid_peak_positions], descending=True)
                ].tolist():
                    if pos not in peak_positions:
                        peak_positions.append(pos)
                        peak_sources[pos] = "fallback_relaxed_nms"
                        if len(peak_positions) == required_peaks:
                            break

            sorted_positions = sorted(peak_positions)
            peak_position_tensor = torch.tensor(
                sorted_positions, device=candidates.device, dtype=torch.long
            )
            peak_frames = candidates[peak_position_tensor]
            peak_sources_sorted = [peak_sources[pos] for pos in sorted_positions]
            selected_raw = raw_row[peak_position_tensor].float()
            selected_smoothed = row[peak_position_tensor].float()
            selected_prominence = prominence[peak_position_tensor].float()
            probe_values["selected_raw"].extend(selected_raw)
            probe_values["selected_smoothed"].extend(selected_smoothed)
            probe_values["selected_prominence"].extend(selected_prominence)
            probe_values["selected_positive_prominence"].extend(
                (selected_prominence > 0).float()
            )
            probe_values["selected_fallback"].extend([
                float(source != "local_max") for source in peak_sources_sorted
            ])
            probe_values["selected_relaxed_nms"].extend([
                float(source == "fallback_relaxed_nms")
                for source in peak_sources_sorted
            ])
            safe_first_pos = margin + 1
            safe_last_pos = row.numel() - 1 - margin
            probe_values["selected_unsafe_edge"].extend([
                float(pos < safe_first_pos or pos > safe_last_pos)
                for pos in sorted_positions
            ])
            probe_values["selected_first_boundary"].extend([
                float(pos == 1) for pos in sorted_positions
            ])
            probe_values["selected_last_boundary"].extend([
                float(pos == row.numel() - 1) for pos in sorted_positions
            ])
            probe_values["peak_frame"].extend(peak_frames.float())
            if peak_frames.numel() > 1:
                probe_values["peak_separation"].extend(
                    (peak_frames[1:] - peak_frames[:-1]).float()
                )

            segment_starts = torch.cat([candidates[:1], peak_frames])
            segment_ends = torch.cat([peak_frames - 1, candidates[-1:]])
            selected = []
            segment_details = []
            segment_count = segment_starts.numel()
            for segment_idx, (start, end) in enumerate(
                zip(segment_starts, segment_ends)
            ):
                # Prefer frames away from a transition boundary; retain the
                # full segment when the guard band would leave it empty.
                guarded_start = start + (margin if segment_idx > 0 else 0)
                guarded_end = end - (
                    margin if segment_idx < segment_count - 1 else 0
                )
                guard_fallback = bool(guarded_start > guarded_end)
                if guarded_start > guarded_end:
                    guarded_start, guarded_end = start, end
                sampled_frame = torch.randint(
                    guarded_start.item(), guarded_end.item() + 1,
                    (1,), device=candidates.device,
                )
                sampled_frame = sampled_frame.squeeze(0)
                selected.append(sampled_frame)
                probe_values["segment_length"].append(
                    float((end - start + 1).item())
                )
                probe_values["guarded_segment_length"].append(
                    float((guarded_end - guarded_start + 1).item())
                )
                probe_values["guard_fallback"].append(float(guard_fallback))
                probe_values["sample_frame"].append(sampled_frame.float())
                probe_values["sample_first_candidate"].append(
                    float(sampled_frame.item() == candidates[0].item())
                )
                probe_values["sample_last_candidate"].append(
                    float(sampled_frame.item() == candidates[-1].item())
                )
                probe_values["sample_boundary_distance"].append(
                    torch.min(torch.abs(sampled_frame - peak_frames)).float()
                )
                segment_details.append({
                    "raw": [int(start.item()), int(end.item())],
                    "guarded": [
                        int(guarded_start.item()), int(guarded_end.item())
                    ],
                    "sample": int(sampled_frame.item()),
                    "guard_fallback": guard_fallback,
                })
            sampled_all.append(torch.stack(selected))

            ranked_valid = valid_peak_positions[
                torch.argsort(rank_score[valid_peak_positions], descending=True)
            ]
            top_positions = ranked_valid[:min(5, ranked_valid.numel())]
            probe_rows.append({
                "candidate_range": [
                    int(candidates[0].item()), int(candidates[-1].item())
                ],
                "local_max_count": int(local_max.numel()),
                "top_candidates": [
                    {
                        "frame": int(candidates[pos].item()),
                        "raw": round(float(raw_row[pos].float().item()), 6),
                        "smooth": round(float(row[pos].float().item()), 6),
                        "prominence": round(float(prominence[pos].float().item()), 6),
                        "is_local_max": bool(local_max_mask[pos].item()),
                    }
                    for pos in top_positions.tolist()
                ],
                "selected_peaks": [
                    {
                        "frame": int(candidates[pos].item()),
                        "source": peak_sources[pos],
                        "raw": round(float(raw_row[pos].float().item()), 6),
                        "smooth": round(float(row[pos].float().item()), 6),
                        "prominence": round(float(prominence[pos].float().item()), 6),
                    }
                    for pos in sorted_positions
                ],
                "segments": segment_details,
            })

        sampled_all = torch.stack(sampled_all, dim=0)
        device = scores.device

        def _stat(values, statistic, default=0.0):
            if not values:
                return torch.tensor(default, device=device, dtype=torch.float32)
            tensors = [
                value.to(device=device, dtype=torch.float32)
                if torch.is_tensor(value)
                else torch.tensor(value, device=device, dtype=torch.float32)
                for value in values
            ]
            stacked = torch.stack(tensors)
            if statistic == "mean":
                return stacked.mean()
            if statistic == "std":
                return stacked.std(unbiased=False)
            if statistic == "min":
                return stacked.min()
            raise ValueError(f"Unsupported probe statistic: {statistic}")

        mean = lambda key: _stat(probe_values[key], "mean")
        std = lambda key: _stat(probe_values[key], "std")
        minimum = lambda key: _stat(probe_values[key], "min")
        candidate_raw_mean = mean("candidate_raw")
        candidate_smoothed_mean = mean("candidate_smoothed")
        selected_raw_mean = mean("selected_raw")
        selected_smoothed_mean = mean("selected_smoothed")
        eps = 1e-8
        self._qwen_frame_sampling_probe = {
            "qwen_image_transition_candidate_raw_score_mean": candidate_raw_mean,
            "qwen_image_transition_candidate_raw_score_std": std("candidate_raw"),
            "qwen_image_transition_candidate_smoothed_score_mean": candidate_smoothed_mean,
            "qwen_image_transition_candidate_smoothed_score_std": std("candidate_smoothed"),
            "qwen_image_transition_peak_raw_score_mean": selected_raw_mean,
            "qwen_image_transition_peak_raw_score_std": std("selected_raw"),
            "qwen_image_transition_peak_smoothed_score_mean": selected_smoothed_mean,
            "qwen_image_transition_peak_smoothed_score_std": std("selected_smoothed"),
            "qwen_image_transition_peak_prominence_mean": mean("selected_prominence"),
            "qwen_image_transition_peak_prominence_std": std("selected_prominence"),
            "qwen_image_transition_peak_positive_prominence_fraction": mean(
                "selected_positive_prominence"
            ),
            "qwen_image_transition_selected_to_candidate_raw_score_ratio": (
                selected_raw_mean / candidate_raw_mean.abs().clamp_min(eps)
            ),
            "qwen_image_transition_selected_to_candidate_smoothed_score_ratio": (
                selected_smoothed_mean
                / candidate_smoothed_mean.abs().clamp_min(eps)
            ),
            "qwen_image_transition_peak_fallback_fraction": mean("selected_fallback"),
            "qwen_image_transition_peak_relaxed_nms_fraction": mean(
                "selected_relaxed_nms"
            ),
            "qwen_image_transition_peak_unsafe_edge_fraction": mean(
                "selected_unsafe_edge"
            ),
            "qwen_image_transition_peak_first_boundary_fraction": mean(
                "selected_first_boundary"
            ),
            "qwen_image_transition_peak_last_boundary_fraction": mean(
                "selected_last_boundary"
            ),
            "qwen_image_transition_local_max_count": mean("local_max_count"),
            "qwen_image_transition_peak_frame_mean": mean("peak_frame"),
            "qwen_image_transition_peak_frame_std": std("peak_frame"),
            "qwen_image_transition_peak_separation_mean": mean("peak_separation"),
            "qwen_image_transition_peak_separation_std": std("peak_separation"),
            "qwen_image_transition_segment_length_mean": mean("segment_length"),
            "qwen_image_transition_segment_length_std": std("segment_length"),
            "qwen_image_transition_segment_length_min": minimum("segment_length"),
            "qwen_image_transition_guarded_segment_length_mean": mean(
                "guarded_segment_length"
            ),
            "qwen_image_transition_guarded_segment_length_std": std(
                "guarded_segment_length"
            ),
            "qwen_image_transition_guarded_segment_length_min": minimum(
                "guarded_segment_length"
            ),
            "qwen_image_transition_guard_fallback_fraction": mean("guard_fallback"),
            "qwen_image_transition_sample_frame_mean": mean("sample_frame"),
            "qwen_image_transition_sample_frame_std": std("sample_frame"),
            "qwen_image_transition_sample_first_candidate_fraction": mean(
                "sample_first_candidate"
            ),
            "qwen_image_transition_sample_last_candidate_fraction": mean(
                "sample_last_candidate"
            ),
            "qwen_image_transition_sample_boundary_distance_mean": mean(
                "sample_boundary_distance"
            ),
            "qwen_image_transition_sample_boundary_distance_std": std(
                "sample_boundary_distance"
            ),
        }
        if self.qwen_image_log_frame_sampling and (
            not dist.is_initialized() or dist.get_rank() == 0
        ):
            anchored = bool(self.qwen_image_include_first_frame)
            print(
                "[Qwen frame sampling probe] "
                f"rows={probe_rows} "
                f"fixed_first_frame={0 if anchored else None}",
                flush=True,
            )
        return sampled_all

    @staticmethod
    def _sample_qwen_image_frames_random_from_candidates(
        candidates: torch.Tensor,
        batch_size: int,
        num_samples: int,
    ) -> torch.Tensor:
        frame_indices = []
        for _ in range(batch_size):
            perm = torch.randperm(candidates.numel(), device=candidates.device)
            frame_indices.append(candidates[perm[:num_samples]])
        return torch.stack(frame_indices, dim=0)

    @staticmethod
    def _sample_qwen_image_frames_temporal_stratified(
        candidates: torch.Tensor,
        batch_size: int,
        num_samples: int,
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Uniformly sample one frame from each contiguous temporal stratum."""
        if num_samples <= 0 or num_samples > candidates.numel():
            raise ValueError(
                "temporal_stratified_random requires 1 <= num_samples <= "
                f"num_candidates, got {num_samples} and {candidates.numel()}"
            )

        # Integer boundaries make segment sizes differ by at most one. For
        # candidates 0..20 and K=3 this is exactly [0..6], [7..13], [14..20].
        boundaries = torch.div(
            torch.arange(
                num_samples + 1,
                device=candidates.device,
                dtype=torch.long,
            ) * candidates.numel(),
            num_samples,
            rounding_mode="floor",
        )
        sampled_batches = []
        strata = []
        for stratum_idx in range(num_samples):
            begin = int(boundaries[stratum_idx].item())
            end = int(boundaries[stratum_idx + 1].item())
            strata.append((
                int(candidates[begin].item()),
                int(candidates[end - 1].item()),
            ))

        for _ in range(batch_size):
            sampled = []
            for stratum_idx in range(num_samples):
                begin = int(boundaries[stratum_idx].item())
                end = int(boundaries[stratum_idx + 1].item())
                candidate_pos = torch.randint(
                    begin, end, (1,), device=candidates.device
                )
                sampled.append(candidates[candidate_pos].squeeze(0))
            sampled_batches.append(torch.stack(sampled))
        return torch.stack(sampled_batches, dim=0), strata

    def _sample_qwen_image_frames(
        self,
        image_or_video: torch.Tensor,
        gradient_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        self._qwen_frame_sampling_probe = {}
        batch_size, num_frame = image_or_video.shape[:2]
        # ``include_first_frame`` is an anchor policy, not merely candidate
        # eligibility: when enabled, frame 0 is always part of the sampled
        # set and the configured frame budget includes that anchored frame.
        force_first_frame = self.qwen_image_include_first_frame and num_frame > 0
        requested_samples = min(self.qwen_image_frames_per_video, num_frame)
        if requested_samples <= 0:
            raise ValueError("qwen_image_teacher.frames_per_video must be >= 1")

        sampling = str(self.qwen_image_frame_sampling).lower().replace("-", "_")
        stratified_sampling = sampling in (
            "temporal_stratified_random",
            "stratified_random",
            "temporal_stratified",
        )
        if stratified_sampling and force_first_frame:
            raise ValueError(
                "temporal_stratified_random treats frame 0 as a random "
                "candidate; it cannot be combined with include_first_frame=true"
            )

        random_first_frame_candidate = (
            self.qwen_image_random_include_first_frame
            and sampling in ("random", "uniform_random")
        )
        start_frame = 0 if (
            random_first_frame_candidate
            or (
                stratified_sampling
                and self.qwen_image_stratified_include_first_frame
            )
        ) else 1
        if start_frame >= num_frame:
            start_frame = 0
        candidates = torch.arange(start_frame, num_frame, device=image_or_video.device)
        num_samples = requested_samples - (1 if force_first_frame else 0)

        if num_samples == 0:
            frame_indices = torch.zeros(
                (batch_size, 1), dtype=torch.long, device=image_or_video.device
            )
        else:
            num_samples = min(num_samples, candidates.numel())

            if sampling in ("random", "uniform_random"):
                frame_indices = self._sample_qwen_image_frames_random_from_candidates(
                    candidates, batch_size, num_samples
                )
            elif stratified_sampling:
                frame_indices, strata = (
                    self._sample_qwen_image_frames_temporal_stratified(
                        candidates, batch_size, num_samples
                    )
                )
                if self.qwen_image_log_frame_sampling and (
                    not dist.is_initialized() or dist.get_rank() == 0
                ):
                    print(
                        "[Qwen frame sampling probe] "
                        f"temporal_strata={strata} "
                        f"segment_samples={frame_indices.tolist()} "
                        "fixed_first_frame=None",
                        flush=True,
                    )
            elif sampling in (
                "difference_peak",
                "difference_peaks",
                "latent_difference",
                "latent_difference_peak",
            ):
                scores = self._compute_qwen_image_frame_difference(
                    image_or_video.detach(), candidates
                )
                frame_indices = self._select_qwen_image_difference_peaks(
                    scores, candidates, num_samples
                )
            elif sampling == "transition_segments":
                scores = self._compute_qwen_image_frame_difference(
                    image_or_video.detach(), candidates
                )
                frame_indices = self._select_qwen_image_transition_segments(
                    scores, candidates, num_samples
                )
            else:
                raise ValueError(
                    "qwen_image_teacher.frame_sampling must be one of "
                    "'random', 'temporal_stratified_random', "
                    "'difference_peak', or 'transition_segments', got "
                    f"{self.qwen_image_frame_sampling!r}"
                )

            if force_first_frame:
                first_frame = torch.zeros(
                    (batch_size, 1), dtype=torch.long, device=image_or_video.device
                )
                frame_indices = torch.cat([first_frame, frame_indices], dim=1)

        gather_index = frame_indices[:, :, None, None, None].expand(
            -1, -1, *image_or_video.shape[2:]
        )
        sampled_latent = torch.gather(image_or_video, dim=1, index=gather_index)

        sampled_mask = None
        if gradient_mask is not None:
            sampled_mask = torch.gather(gradient_mask, dim=1, index=gather_index)
        return sampled_latent, sampled_mask, frame_indices

    def _get_qwen_image_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
    ) -> torch.Tensor:
        sampling = str(self.qwen_image_timestep_sampling).lower().replace("_", "-")
        if sampling in ("random", "uniform"):
            return self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                1,
                uniform_timestep=True,
            )

        if max_timestep <= min_timestep:
            raise ValueError(
                f"Invalid Qwen timestep range: [{min_timestep}, {max_timestep})"
            )

        if sampling in ("u-form", "uform", "u-shaped", "u-shape"):
            beta = torch.distributions.Beta(
                torch.tensor(0.5, device=self.device),
                torch.tensor(0.5, device=self.device),
            )
            ratio = beta.sample((batch_size, 1))
        elif sampling in ("high-frequency", "highfreq", "late", "late-denoise"):
            # Late denoising steps have smaller timestep values and emphasize
            # low-noise image detail/high-frequency visual corrections.
            ratio = torch.rand(batch_size, 1, device=self.device).pow(2.0)
        else:
            raise ValueError(
                "qwen_image_teacher.timestep_sampling must be one of "
                "'random', 'u-form', or 'high-frequency', got "
                f"{self.qwen_image_timestep_sampling!r}"
            )

        timestep = min_timestep + torch.floor(ratio * (max_timestep - min_timestep))
        timestep = timestep.to(dtype=torch.long).clamp(min_timestep, max_timestep - 1)
        return timestep.repeat(1, num_frame)

    def _get_qwen_decoupled_timesteps(
        self,
        generation_timestep: int,
        batch_size: int,
        num_frame: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample independent MD and CA timesteps for Qwen Decoupled DMD.

        MD remains global uniform random. The Decoupled DMD paper uses 0 for
        pure noise and 1 for clean data, while Wan uses the reverse convention
        (T for pure noise and 0 for clean data). Therefore paper tau_CA > t is
        sampled here as Wan tau_CA < t. The timestep shift is applied only
        after both schedules have been sampled.
        """
        md_timestep = self._get_timestep(
            self.min_score_timestep,
            self.num_train_timestep,
            batch_size,
            num_frame,
            1,
            uniform_timestep=True,
        )

        generation_timestep = int(generation_timestep)
        ca_max_timestep = min(self.num_train_timestep, generation_timestep)
        if ca_max_timestep <= self.min_score_timestep:
            raise ValueError(
                "decoupled_dmd_for_qwen requires the Wan generator input "
                "timestep to exceed min_score_timestep so paper tau_CA > t "
                "can be sampled in the reversed Wan coordinate, got "
                f"t={generation_timestep} and "
                f"min_score_timestep={self.min_score_timestep}"
            )
        ca_timestep = self._get_timestep(
            self.min_score_timestep,
            ca_max_timestep,
            batch_size,
            num_frame,
            1,
            uniform_timestep=True,
        )
        return md_timestep, ca_timestep

    def _shift_and_clamp_qwen_timestep(
        self,
        timestep: torch.Tensor,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        if self.qwen_image_timestep_schedule in (
            "qwen_dynamic",
            "qwen_native",
        ):
            return map_qwen_raw_timestep(
                timestep,
                latent_height=latent_height,
                latent_width=latent_width,
                scheduler_config=self.qwen_image_scheduler_config,
                num_train_timesteps=self.num_train_timestep,
                reference_num_inference_steps=(
                    self.qwen_image_scheduler_reference_steps
                ),
                min_effective_timestep=self.min_step,
                max_effective_timestep=self.max_step,
            )

        if self.timestep_shift > 1:
            timestep = self.timestep_shift * (
                timestep / 1000
            ) / (
                1 + (self.timestep_shift - 1) * (timestep / 1000)
            ) * 1000
        return timestep.clamp(self.min_step, self.max_step)

    def _prepare_qwen_image_latents(
        self,
        image_or_video: torch.Tensor,
        gradient_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """Sample video chunks and map them into Qwen single-frame latents."""
        sampled_latent, sampled_mask, frame_indices = self._sample_qwen_image_frames(
            image_or_video, gradient_mask
        )
        frame_offsets = None
        if self.qwen_frame_latent_adapter is not None:
            # Continuation index k uses a random physical-frame offset. The
            # independent first latent is already an image latent and bypasses
            # the adapter, with -1 retained as its logging sentinel.
            frame_offsets = torch.randint(
                0,
                self.qwen_frame_latent_adapter.config.num_frame_offsets,
                frame_indices.shape,
                device=frame_indices.device,
                dtype=torch.long,
            )
            frame_offsets = torch.where(
                frame_indices == 0,
                torch.full_like(frame_offsets, -1),
                frame_offsets,
            )
            sampled_latent = self.qwen_frame_latent_adapter.extract_from_video(
                image_or_video,
                chunk_indices=frame_indices,
                frame_offsets=frame_offsets,
            )
        return sampled_latent, sampled_mask, frame_indices, frame_offsets

    def compute_qwen_image_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        text_prompts: List[str],
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        video_wan_grad: Optional[torch.Tensor] = None,
        qwen_loss_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, dict]:
        if not self.qwen_image_teacher_enabled:
            zero = image_or_video.sum() * 0.0
            return zero, {}

        sampled_latent, sampled_mask, frame_indices, frame_offsets = (
            self._prepare_qwen_image_latents(image_or_video, gradient_mask)
        )
        original_latent = sampled_latent
        batch_size, num_frame = sampled_latent.shape[:2]

        with torch.no_grad():
            qwen_md_grad = None
            qwen_ca_grad = None
            ca_timestep = None
            if self.decoupled_dmd_for_qwen:
                # The paper's t is the timestep of generator input z_t, which
                # is ``denoised_timestep_from`` here. Since Wan's timestep
                # direction is reversed, paper tau_CA > t means the sampled
                # Wan CA timestep is numerically smaller than this value.
                generation_timestep = (
                    denoised_timestep_from
                    if denoised_timestep_from is not None
                    else self.num_train_timestep
                )
                timestep, ca_timestep = self._get_qwen_decoupled_timesteps(
                    generation_timestep,
                    batch_size,
                    num_frame,
                )
                raw_timestep = timestep
                raw_ca_timestep = ca_timestep
                timestep = self._shift_and_clamp_qwen_timestep(
                    timestep,
                    sampled_latent.shape[-2],
                    sampled_latent.shape[-1],
                )
                ca_timestep = self._shift_and_clamp_qwen_timestep(
                    ca_timestep,
                    sampled_latent.shape[-2],
                    sampled_latent.shape[-1],
                )

                md_noise = torch.randn_like(sampled_latent)
                md_noisy_latent = self.scheduler.add_noise(
                    sampled_latent.flatten(0, 1),
                    md_noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).detach().unflatten(0, (batch_size, num_frame))
                ca_noise = torch.randn_like(sampled_latent)
                ca_noisy_latent = self.scheduler.add_noise(
                    sampled_latent.flatten(0, 1),
                    ca_noise.flatten(0, 1),
                    ca_timestep.flatten(0, 1),
                ).detach().unflatten(0, (batch_size, num_frame))

                pred_fake_cond = self._compute_qwen_image_fake_score(
                    noisy_latent=md_noisy_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                pred_qwen_md_cond = self.qwen_image_score(
                    md_noisy_latent,
                    timestep,
                    text_prompts=text_prompts,
                    negative_prompt=self.qwen_image_negative_prompt,
                    guidance_scale=1.0,
                )
                pred_qwen_ca_cond, pred_qwen_ca_uncond = self.qwen_image_score(
                    ca_noisy_latent,
                    ca_timestep,
                    text_prompts=text_prompts,
                    negative_prompt=self.qwen_image_negative_prompt,
                    guidance_scale=self.qwen_image_guidance_scale,
                    return_conditional_and_unconditional_predictions=True,
                )

                md_normalizer = self._qwen_per_frame_normalizer(
                    original_latent, pred_qwen_md_cond
                )
                ca_normalizer = self._qwen_per_frame_normalizer(
                    original_latent, pred_qwen_ca_cond
                )
                qwen_md_grad = torch.nan_to_num(
                    (pred_fake_cond - pred_qwen_md_cond) / md_normalizer
                )
                qwen_ca_grad = torch.nan_to_num(
                    (self.qwen_image_guidance_scale - 1.0)
                    * (pred_qwen_ca_uncond - pred_qwen_ca_cond)
                    / ca_normalizer
                )
                qwen_dmd_grad = torch.nan_to_num(qwen_md_grad + qwen_ca_grad)
                noisy_latent = md_noisy_latent
            else:
                min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
                max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
                timestep = self._get_qwen_image_timestep(
                    min_timestep,
                    max_timestep,
                    batch_size,
                    num_frame,
                )
                raw_timestep = timestep
                raw_ca_timestep = None
                timestep = self._shift_and_clamp_qwen_timestep(
                    timestep,
                    sampled_latent.shape[-2],
                    sampled_latent.shape[-1],
                )

                noise = torch.randn_like(sampled_latent)
                noisy_latent = self.scheduler.add_noise(
                    sampled_latent.flatten(0, 1),
                    noise.flatten(0, 1),
                    timestep.flatten(0, 1),
                ).detach().unflatten(0, (batch_size, num_frame))

                pred_fake_cond = self._compute_qwen_image_fake_score(
                    noisy_latent=noisy_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                pred_real = self.qwen_image_score(
                    noisy_latent,
                    timestep,
                    text_prompts=text_prompts,
                    negative_prompt=self.qwen_image_negative_prompt,
                    guidance_scale=self.qwen_image_guidance_scale,
                )

                qwen_grad = pred_fake_cond - pred_real
                qwen_normalizer = self._qwen_per_frame_normalizer(
                    original_latent, pred_real
                )
                qwen_dmd_grad = torch.nan_to_num(qwen_grad / qwen_normalizer)
            grad = qwen_dmd_grad

            grad = torch.nan_to_num(grad)

        qwen_loss = self._dmd_loss_from_grad(original_latent, grad, sampled_mask)
        qwen_component_loss = self._dmd_loss_from_grad(
            original_latent, qwen_dmd_grad, sampled_mask
        )

        qwen_log_dict = {
            "legacy__qwen-dmd_loss": qwen_loss.detach(),
            "qwen_image_qwen_dmd_loss": qwen_component_loss.detach(),
            "legacy__qwen-dmd_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "qwen_image_qwen_dmd_gradient_norm": torch.mean(
                torch.abs(qwen_dmd_grad)
            ).detach(),
            "qwen_image_raw_timestep": raw_timestep.detach().float(),
            "qwen_image_timestep": timestep.detach(),
            "qwen_image_effective_sigma": (
                timestep.detach() / self.num_train_timestep
            ),
            "qwen_image_frame_indices": frame_indices.detach().float(),
        }
        if frame_offsets is not None:
            qwen_log_dict["qwen_image_frame_offsets"] = (
                frame_offsets.detach().float()
            )
        for frame_slot in range(frame_indices.shape[1]):
            qwen_log_dict[f"qwen_image_frame_idx_{frame_slot}"] = (
                frame_indices[:, frame_slot].float().mean().detach()
            )
            if frame_offsets is not None:
                qwen_log_dict[f"qwen_image_frame_offset_{frame_slot}"] = (
                    frame_offsets[:, frame_slot].float().mean().detach()
                )
        qwen_log_dict.update(getattr(self, "_qwen_frame_sampling_probe", {}))
        if qwen_md_grad is not None:
            qwen_log_dict.update({
                "qwen_image_raw_ca_timestep": raw_ca_timestep.detach().float(),
                "qwen_image_md_dmd_loss": self._dmd_loss_from_grad(
                    original_latent, qwen_md_grad, sampled_mask
                ).detach(),
                "qwen_image_ca_dmd_loss": self._dmd_loss_from_grad(
                    original_latent, qwen_ca_grad, sampled_mask
                ).detach(),
                "qwen_image_md_gradient_norm": torch.mean(
                    torch.abs(qwen_md_grad)
                ).detach(),
                "qwen_image_ca_gradient_norm": torch.mean(
                    torch.abs(qwen_ca_grad)
                ).detach(),
                "qwen_image_ca_timestep": ca_timestep.detach(),
                "qwen_image_probe_ca_timestep_mean": ca_timestep.float().mean().detach(),
                "qwen_image_probe_ca_timestep_std": ca_timestep.float().std(
                    unbiased=False
                ).detach(),
                "qwen_image_probe_ca_timestep_min": ca_timestep.float().min().detach(),
                "qwen_image_probe_ca_timestep_max": ca_timestep.float().max().detach(),
            })
        # Adapter output gradients live in Qwen frame coordinates and flow to
        # both z[k-1] and z[k]. They cannot be compared to a single gathered
        # Wan-frame gradient with the direct-path cosine diagnostic.
        if video_wan_grad is not None and frame_offsets is None:
            qwen_log_dict.update(
                self._qwen_video_wan_gradient_logs(
                    qwen_grad=qwen_dmd_grad,
                    video_wan_grad=video_wan_grad,
                    frame_indices=frame_indices,
                    qwen_loss_weight=qwen_loss_weight,
                    qwen_mask=sampled_mask,
                    video_mask=gradient_mask,
                )
            )
        return qwen_loss, qwen_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        text_prompts: Optional[List[str]] = None,
        current_step: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict, video_wan_grad = (
            self.compute_distribution_matching_loss(
                image_or_video=pred_image,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                gradient_mask=gradient_mask,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
        )
        dmd_log_dict["video_dmd_loss"] = dmd_loss.detach()

        if self.qwen_image_teacher_enabled and text_prompts is not None:
            warmup = self.qwen_image_warmup_steps
            if warmup > 0:
                qwen_weight = self.qwen_image_loss_weight * min(1.0, current_step / warmup)
            else:
                qwen_weight = self.qwen_image_loss_weight
            if qwen_weight <= 0:
                dmd_log_dict["qwen_image_loss_weight"] = torch.tensor(
                    qwen_weight, device=pred_image.device
                )
                return dmd_loss, dmd_log_dict

            qwen_loss, qwen_log_dict = self.compute_qwen_image_distribution_matching_loss(
                image_or_video=pred_image,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                text_prompts=text_prompts,
                gradient_mask=gradient_mask,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
                video_wan_grad=video_wan_grad,
                qwen_loss_weight=qwen_weight,
            )
            dmd_loss = dmd_loss + qwen_weight * qwen_loss
            dmd_log_dict.update(qwen_log_dict)
            dmd_log_dict["qwen_image_loss_weight"] = torch.tensor(
                qwen_weight, device=pred_image.device
            )

        return dmd_loss, dmd_log_dict

    def compute_qwen_image_fake_critic_loss(
        self,
        generated_image: torch.Tensor,
        conditional_dict: dict,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """Train the dedicated fake critic on adapter-space image marginals."""
        if not self.qwen_image_fake_critic_enabled:
            return generated_image.sum() * 0.0, {}

        with torch.no_grad():
            sampled_latent, _, frame_indices, frame_offsets = (
                self._prepare_qwen_image_latents(generated_image.detach())
            )
            batch_size, num_frame = sampled_latent.shape[:2]
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            raw_timestep = self._get_qwen_image_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
            )
            timestep = self._shift_and_clamp_qwen_timestep(
                raw_timestep,
                sampled_latent.shape[-2],
                sampled_latent.shape[-1],
            )
            noise = torch.randn_like(sampled_latent)
            noisy_latent = self.scheduler.add_noise(
                sampled_latent.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))

        flow_pred, x0_pred = self._compute_independent_frame_fake_predictions(
            noisy_latent=noisy_latent,
            timestep=timestep,
            conditional_dict=conditional_dict,
            score_model=self.qwen_image_fake_score,
        )
        if self.args.denoising_loss_type == "flow":
            frame_error = (
                flow_pred - (noise - sampled_latent)
            ).square().mean(dim=[2, 3, 4])
        elif self.args.denoising_loss_type == "x0":
            frame_error = (
                x0_pred - sampled_latent
            ).square().mean(dim=[2, 3, 4])
        else:
            raise ValueError(
                "The Qwen image fake critic supports denoising_loss_type "
                f"'flow' or 'x0', got {self.args.denoising_loss_type!r}"
            )
        fake_critic_loss = frame_error.mean()

        def masked_mean(mask: torch.Tensor) -> torch.Tensor:
            selected = frame_error[mask]
            if selected.numel() == 0:
                return frame_error.detach().sum() * 0.0
            return selected.detach().mean()

        log_dict = {
            "qwen_image_fake_critic_loss": fake_critic_loss.detach(),
            "qwen_image_fake_critic_raw_timestep": raw_timestep.detach().float(),
            "qwen_image_fake_critic_timestep": timestep.detach(),
            "qwen_image_fake_critic_first_frame_loss": masked_mean(
                frame_indices == 0
            ),
            "qwen_image_fake_critic_continuation_loss": masked_mean(
                frame_indices > 0
            ),
        }
        if frame_offsets is not None:
            for offset in range(
                self.qwen_frame_latent_adapter.config.num_frame_offsets
            ):
                log_dict[
                    f"qwen_image_fake_critic_offset_{offset}_loss"
                ] = masked_mean(frame_offsets == offset)
        return fake_critic_loss, log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        qwen_fake_critic_loss, qwen_fake_critic_log_dict = (
            self.compute_qwen_image_fake_critic_loss(
                generated_image=generated_image,
                conditional_dict=conditional_dict,
                denoised_timestep_from=denoised_timestep_from,
                denoised_timestep_to=denoised_timestep_to,
            )
        )
        total_critic_loss = denoising_loss + (
            self.qwen_image_fake_critic_loss_weight * qwen_fake_critic_loss
        )

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "video_critic_loss": denoising_loss.detach(),
        }
        critic_log_dict.update(qwen_fake_critic_log_dict)

        return total_critic_loss, critic_log_dict

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor, tf=False, causal = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        # Step 1: Randomly choose a timestep for each frame
        uniform_timestep = tf or (not causal)
        print(f'uniform_timestep is {uniform_timestep}')
        index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=uniform_timestep
        )
        print(f'before self._process_timestep(index), index is {index}')
        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent, dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1).expand(
                -1, -1, -1, num_channels, height, width).to(self.device)
        ).squeeze(1)

        timestep = self.denoising_step_list[index].to(self.device)
        print(f'index is {index}, timestep is {timestep}')


        return noisy_input, timestep
