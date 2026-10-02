"""Qwen-Image resolution-dependent timestep mapping for DMD training."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

import torch


def load_qwen_scheduler_config(model_path: str | Path) -> dict:
    """Load the scheduler shipped with a local Qwen-Image checkpoint."""
    config_path = Path(model_path) / "scheduler" / "scheduler_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Qwen scheduler config does not exist: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("_class_name") != "FlowMatchEulerDiscreteScheduler":
        raise ValueError(
            "Qwen DMD requires FlowMatchEulerDiscreteScheduler, got "
            f"{config.get('_class_name')!r}"
        )
    return config


def _dynamic_shift(
    raw_sigma: torch.Tensor, mu: float, time_shift_type: str
) -> torch.Tensor:
    if time_shift_type == "exponential":
        shift = math.exp(mu)
    elif time_shift_type == "linear":
        shift = mu
    else:
        raise ValueError(
            "Qwen scheduler time_shift_type must be 'exponential' or "
            f"'linear', got {time_shift_type!r}"
        )
    return shift * raw_sigma / (1.0 + (shift - 1.0) * raw_sigma)


def map_qwen_raw_timestep(
    raw_timestep: torch.Tensor,
    *,
    latent_height: int,
    latent_width: int,
    scheduler_config: Mapping[str, object],
    num_train_timesteps: int = 1000,
    reference_num_inference_steps: int = 50,
    min_effective_timestep: float = 20.0,
    max_effective_timestep: float = 980.0,
) -> torch.Tensor:
    """Map raw timestep coordinates through Qwen's native dynamic shift.

    Qwen packs each image latent into 2x2 spatial patches. The packed image
    sequence length determines ``mu``, which in turn determines the dynamic
    shift. When the checkpoint config specifies ``shift_terminal``, use the
    final raw sigma of its reference inference grid to reproduce Diffusers'
    schedule-wide terminal stretch for arbitrary sampled raw timesteps.
    """
    if latent_height % 2 or latent_width % 2:
        raise ValueError(
            "Qwen latent height and width must be even, got "
            f"{latent_height}x{latent_width}"
        )
    if num_train_timesteps <= 0 or reference_num_inference_steps <= 0:
        raise ValueError("Timestep counts must be positive")

    raw_sigma = raw_timestep.to(dtype=torch.float32) / num_train_timesteps
    image_seq_len = (latent_height // 2) * (latent_width // 2)

    if bool(scheduler_config.get("use_dynamic_shifting", False)):
        base_seq_len = int(scheduler_config.get("base_image_seq_len", 256))
        max_seq_len = int(scheduler_config.get("max_image_seq_len", 4096))
        base_shift = float(scheduler_config.get("base_shift", 0.5))
        max_shift = float(scheduler_config.get("max_shift", 1.15))
        if max_seq_len == base_seq_len:
            raise ValueError("Qwen scheduler sequence-length anchors must differ")
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        mu = image_seq_len * slope + intercept
        time_shift_type = str(
            scheduler_config.get("time_shift_type", "exponential")
        )
        effective_sigma = _dynamic_shift(raw_sigma, mu, time_shift_type)

        shift_terminal = scheduler_config.get("shift_terminal")
        if shift_terminal is not None:
            terminal = float(shift_terminal)
            reference_raw_terminal = raw_sigma.new_tensor(
                1.0 / reference_num_inference_steps
            )
            shifted_reference_terminal = _dynamic_shift(
                reference_raw_terminal, mu, time_shift_type
            )
            scale_factor = (1.0 - shifted_reference_terminal) / (1.0 - terminal)
            effective_sigma = 1.0 - (1.0 - effective_sigma) / scale_factor
    else:
        shift = float(scheduler_config.get("shift", 1.0))
        effective_sigma = (
            shift * raw_sigma / (1.0 + (shift - 1.0) * raw_sigma)
        )

    effective_timestep = effective_sigma * num_train_timesteps
    return effective_timestep.clamp(
        min=min_effective_timestep, max=max_effective_timestep
    )
