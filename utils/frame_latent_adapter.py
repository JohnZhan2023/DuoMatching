"""LatentBridge integration with direct supervision for the independent first frame."""
import torch
from latent_bridge.model import (
    FrameLatentAdapter as SpatialFrameLatentAdapter,
    FrameLatentAdapterConfig,
)


class FrameLatentAdapter(SpatialFrameLatentAdapter):
    """Qwen bridge with a bypass for Wan's independent first-frame latent."""

    def extract_from_video(
        self,
        video_latents: torch.Tensor,
        chunk_indices: torch.Tensor,
        frame_offsets: torch.Tensor,
    ) -> torch.Tensor:
        """Extract K Qwen frame latents from Wan latents shaped [B,F,C,H,W].

        Index 0 is the independent first-frame latent and bypasses the adapter.
        Continuation indices use the learned adapter with their frame offsets.
        """
        if video_latents.ndim != 5:
            raise ValueError("video_latents must have shape [B, F, C, H, W]")
        if chunk_indices.shape != frame_offsets.shape or chunk_indices.ndim != 2:
            raise ValueError("chunk_indices and frame_offsets must have shape [B, K]")
        if torch.any(chunk_indices < 0) or torch.any(
            chunk_indices >= video_latents.shape[1]
        ):
            raise ValueError("chunk_indices must be in [0, F-1]")
        continuation_offsets = frame_offsets[chunk_indices > 0]
        if torch.any(continuation_offsets < 0) or torch.any(
            continuation_offsets >= self.config.num_frame_offsets
        ):
            raise ValueError("continuation frame offsets must be in [0, 3]")

        if chunk_indices.shape[0] != video_latents.shape[0]:
            raise ValueError("chunk index batch does not match video_latents")
        batch_size, num_samples = chunk_indices.shape
        batch_indices = torch.arange(
            batch_size, device=video_latents.device
        )[:, None].expand(batch_size, num_samples)
        safe_chunk_indices = chunk_indices.clamp_min(1).clamp_max(video_latents.shape[1] - 1)
        previous = video_latents[batch_indices, (safe_chunk_indices - 1).clamp_min(0)]
        current = video_latents[batch_indices, safe_chunk_indices]
        adapted = self(
            previous.flatten(0, 1),
            current.flatten(0, 1),
            torch.where(chunk_indices == 0, 0, frame_offsets).flatten(),
        ).unflatten(0, (batch_size, num_samples))
        direct = video_latents[batch_indices, chunk_indices]
        first_frame_mask = (chunk_indices == 0)[..., None, None, None]
        return torch.where(first_frame_mask, direct, adapted)
