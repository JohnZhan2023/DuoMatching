from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, scale)


def distributed_schedule_steps(
    warmup_steps: int,
    total_steps: int,
    num_processes: int,
) -> tuple[int, int]:
    """Convert global steps to AcceleratedScheduler optimizer-step units."""
    if warmup_steps < 0 or total_steps < 1 or num_processes < 1:
        raise ValueError("invalid scheduler step configuration")
    return warmup_steps * num_processes, total_steps * num_processes
