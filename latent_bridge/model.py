from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FrameLatentAdapterConfig:
    """Residual Wan-to-Qwen LatentBridge architecture."""

    latent_channels: int = 16
    hidden_channels: int = 256
    num_res_blocks: int = 8
    frame_embed_dim: int = 256
    num_frame_offsets: int = 4
    groups: int = 32
    residual_output: bool = True

    def validate(self) -> None:
        if self.latent_channels < 1:
            raise ValueError("latent_channels must be positive")
        if self.hidden_channels < 1 or self.frame_embed_dim < 1:
            raise ValueError("hidden_channels and frame_embed_dim must be positive")
        if self.num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        if self.num_frame_offsets != 4:
            raise ValueError("Wan continuation chunks require exactly four frame offsets")
        if self.groups < 1 or self.hidden_channels % self.groups:
            raise ValueError("groups must be positive and divide hidden_channels")


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, embed_dim: int, groups: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels, eps=1e-6)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, channels * 2),
        )

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, hidden: torch.Tensor, frame_embedding: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.conv1(F.silu(self.norm1(hidden)))
        scale, shift = self.to_scale_shift(frame_embedding).chunk(2, dim=-1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        hidden = self.conv2(F.silu(hidden))
        return residual + hidden


class FrameLatentAdapter(nn.Module):
    """
    Map a causal Wan latent pair and an in-chunk offset to one Qwen image latent.

    The previous latent is necessary because a continuation latent is encoded with
    causal context. The output is initialized as a residual correction to the
    current latent, which is useful because Wan and Qwen use the same normalized
    16-channel coordinate system.
    """

    config_name = "adapter_config.json"
    weights_name = "adapter.pt"

    def __init__(self, config: FrameLatentAdapterConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.frame_embedding = nn.Embedding(
            config.num_frame_offsets,
            config.frame_embed_dim,
        )
        self.frame_mlp = nn.Sequential(
            nn.Linear(config.frame_embed_dim, config.frame_embed_dim),
            nn.SiLU(),
            nn.Linear(config.frame_embed_dim, config.frame_embed_dim),
        )
        self.input_proj = nn.Conv2d(
            config.latent_channels * 2,
            config.hidden_channels,
            3,
            padding=1,
        )
        self.blocks = nn.ModuleList(
            [
                FiLMResidualBlock(
                    config.hidden_channels,
                    config.frame_embed_dim,
                    config.groups,
                )
                for _ in range(config.num_res_blocks)
            ]
        )
        self.output_norm = nn.GroupNorm(
            config.groups,
            config.hidden_channels,
            eps=1e-6,
        )
        self.output_proj = nn.Conv2d(
            config.hidden_channels,
            config.latent_channels,
            3,
            padding=1,
        )
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        previous_latent: torch.Tensor,
        current_latent: torch.Tensor,
        frame_offset: torch.Tensor,
    ) -> torch.Tensor:
        if previous_latent.shape != current_latent.shape:
            raise ValueError(
                "previous_latent and current_latent must have identical shapes, "
                f"got {tuple(previous_latent.shape)} and {tuple(current_latent.shape)}"
            )
        if previous_latent.ndim != 4:
            raise ValueError("latents must have shape [B, C, H, W]")
        if frame_offset.shape != (previous_latent.shape[0],):
            raise ValueError("frame_offset must have shape [B]")
        if frame_offset.dtype != torch.long:
            frame_offset = frame_offset.long()
        if torch.any(frame_offset < 0) or torch.any(
            frame_offset >= self.config.num_frame_offsets
        ):
            raise ValueError("frame_offset values must be in [0, 3]")

        frame_embedding = self.frame_mlp(self.frame_embedding(frame_offset))
        hidden = self.input_proj(torch.cat([previous_latent, current_latent], dim=1))
        for block in self.blocks:
            hidden = block(hidden, frame_embedding)
        correction = self.output_proj(F.silu(self.output_norm(hidden)))
        if self.config.residual_output:
            return current_latent + correction
        return correction

    def extract_from_video(
        self,
        video_latents: torch.Tensor,
        chunk_indices: torch.Tensor,
        frame_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Extract K frame latents from `[B,F,C,H,W]` Wan latent sequences."""
        if video_latents.ndim != 5:
            raise ValueError("video_latents must have shape [B, F, C, H, W]")
        if chunk_indices.shape != frame_offsets.shape or chunk_indices.ndim != 2:
            raise ValueError("chunk_indices and frame_offsets must both have shape [B, K]")
        batch_size, num_chunks = chunk_indices.shape
        if batch_size != video_latents.shape[0]:
            raise ValueError("chunk index batch does not match video_latents")
        if torch.any(chunk_indices < 1) or torch.any(
            chunk_indices >= video_latents.shape[1]
        ):
            raise ValueError("chunk indices must select continuation latents in [1, F-1]")

        batch_indices = torch.arange(
            batch_size,
            device=video_latents.device,
        )[:, None].expand(batch_size, num_chunks)
        previous = video_latents[batch_indices, chunk_indices - 1]
        current = video_latents[batch_indices, chunk_indices]
        output = self(
            previous.flatten(0, 1),
            current.flatten(0, 1),
            frame_offsets.flatten(),
        )
        return output.unflatten(0, (batch_size, num_chunks))

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / self.config_name).write_text(
            json.dumps(asdict(self.config), indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.state_dict(), output_dir / self.weights_name)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "FrameLatentAdapter":
        model_dir = Path(model_dir)
        config = FrameLatentAdapterConfig(
            **json.loads((model_dir / cls.config_name).read_text(encoding="utf-8"))
        )
        model = cls(config)
        state_dict = torch.load(
            model_dir / cls.weights_name,
            map_location=map_location,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model
