import os
import re
import math
from typing import List

import torch
import torch.distributed as dist
import wandb
from einops import rearrange
from torchvision.io import write_video

from pipeline import CausalInferencePipeline


def _read_prompts(prompt_path: str) -> List[str]:
    with open(prompt_path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _safe_filename(text: str, max_length: int = 80) -> str:
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text[:max_length].strip("._-") or "prompt"


class LogStepInferenceHook:
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config
        self.pipeline = None

    def _build_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline

        self.pipeline = CausalInferencePipeline(
            self.config,
            device=self.trainer.device,
            generator=self.trainer.model.generator,
            text_encoder=self.trainer.model.text_encoder,
            vae=self.trainer.model.vae,
        )
        self.pipeline.eval()
        return self.pipeline

    @torch.no_grad()
    def _snapshot_generator_params(self):
        return {
            name: param.detach().clone().cpu()
            for name, param in self.trainer.model.generator.module.named_parameters()
        }

    @torch.no_grad()
    def _restore_generator_params(self, snapshot):
        for name, param in self.trainer.model.generator.module.named_parameters():
            if name in snapshot:
                param.data.copy_(snapshot[name].to(dtype=param.dtype, device=param.device))

    @torch.no_grad()
    def __call__(self, step: int):
        prompt_path = getattr(self.config, "log_inference_prompt_path", "prompts/demos.txt")
        prompts = _read_prompts(prompt_path)
        if not prompts:
            if self.trainer.is_main_process:
                print(f"No prompts found at {prompt_path}; skip log-step inference.")
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        num_rounds = math.ceil(len(prompts) / world_size)

        output_dir = os.path.join(
            self.trainer.output_path,
            "log_inference",
            f"step_{step:06d}",
        )
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        pipeline = self._build_pipeline()
        num_output_frames = getattr(self.config, "log_inference_num_output_frames", None)
        if num_output_frames is None:
            num_output_frames = self.config.image_or_video_shape[1]
        latent_channels, latent_height, latent_width = self.config.image_or_video_shape[2:]
        fps = getattr(self.config, "log_inference_fps", 16)
        seed = int(getattr(self.config, "log_inference_seed", self.config.seed)) + step

        saved_paths = []
        was_training = pipeline.training
        pipeline.eval()

        raw_generator_snapshot = None
        ema = getattr(self.trainer, "generator_ema", None)
        if ema is not None:
            raw_generator_snapshot = self._snapshot_generator_params()
            ema.copy_to(self.trainer.model.generator)
            if self.trainer.is_main_process:
                print("Log-step inference using generator EMA weights.")
        elif self.trainer.is_main_process:
            print("EMA is not available; log-step inference using current generator weights.")

        try:
            for round_idx in range(num_rounds):
                prompt_index = round_idx * world_size + rank
                has_real_prompt = prompt_index < len(prompts)
                prompt = prompts[prompt_index] if has_real_prompt else prompts[-1]
                generator = torch.Generator(device=torch.device("cuda", self.trainer.device))
                generator.manual_seed(seed + prompt_index)
                sampled_noise = torch.randn(
                    [1, num_output_frames, latent_channels, latent_height, latent_width],
                    device=self.trainer.device,
                    dtype=self.trainer.dtype,
                    generator=generator,
                )

                video, _ = pipeline.inference(
                    noise=sampled_noise,
                    text_prompts=[prompt],
                    return_latents=True,
                )
                video = (255.0 * rearrange(video, "b t c h w -> b t h w c")).clamp(0, 255)
                video = video.cpu().to(torch.uint8)
                if has_real_prompt:
                    output_path = os.path.join(
                        output_dir,
                        f"{_safe_filename(prompt)}.mp4",
                    )
                    write_video(output_path, video[0], fps=fps)
                    saved_paths.append((prompt_index, prompt, output_path))

                if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                    pipeline.vae.model.clear_cache()
                torch.cuda.empty_cache()
        finally:
            if raw_generator_snapshot is not None:
                self._restore_generator_params(raw_generator_snapshot)
                del raw_generator_snapshot

            if was_training:
                pipeline.train()

        gathered = [None for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather_object(gathered, saved_paths)
            dist.barrier()
        else:
            gathered = [saved_paths]

        if self.trainer.is_main_process:
            all_outputs = [item for rank_outputs in gathered for item in rank_outputs]
            all_outputs.sort(key=lambda item: item[0])
            print(f"Log-step inference saved {len(all_outputs)} videos to {output_dir}")

            max_wandb_videos = int(getattr(self.config, "log_inference_wandb_max_videos", 3))
            if (not self.trainer.disable_wandb) and max_wandb_videos > 0:
                wandb_payload = {
                    "log_inference/num_videos": len(all_outputs),
                }
                for prompt_index, prompt, path in all_outputs[:max_wandb_videos]:
                    wandb_payload[f"log_inference/video_{prompt_index:04d}"] = wandb.Video(
                        path,
                        caption=prompt,
                        fps=fps,
                        format="mp4",
                    )
                wandb.log(wandb_payload, step=step)

        if dist.is_initialized():
            dist.barrier()
