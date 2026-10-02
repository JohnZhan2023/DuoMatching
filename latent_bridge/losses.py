from __future__ import annotations

import torch
import torch.nn.functional as F


def spatial_gradient(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        tensor[:, :, 1:, :] - tensor[:, :, :-1, :],
        tensor[:, :, :, 1:] - tensor[:, :, :, :-1],
    )


def adapter_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    l1_weight: float = 1.0,
    cosine_weight: float = 0.1,
    gradient_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")

    # Exclude non-finite cached targets before computing reconstruction losses.
    valid = torch.isfinite(prediction).flatten(1).all(1)
    valid &= torch.isfinite(target).flatten(1).all(1)
    if valid.any():
        prediction_for_loss = prediction[valid]
        target_for_loss = target[valid]
    else:
        # Keep a differentiable zero so every DDP rank still participates in
        # backward even if its local batch contains no usable samples.
        zero = prediction.float().sum() * 0.0
        return zero, {
            "loss": zero.detach(),
            "l1": zero.detach(),
            "cosine": zero.detach(),
            "gradient": zero.detach(),
            "valid_fraction": valid.float().mean().detach(),
        }

    prediction_float = prediction_for_loss.float()
    target_float = target_for_loss.float()
    l1 = F.l1_loss(prediction_float, target_float)
    cosine = 1 - F.cosine_similarity(
        prediction_float.flatten(1),
        target_float.flatten(1),
        dim=1,
        eps=1e-8,
    ).mean()
    pred_dy, pred_dx = spatial_gradient(prediction_float)
    target_dy, target_dx = spatial_gradient(target_float)
    gradient = 0.5 * (
        F.l1_loss(pred_dy, target_dy) + F.l1_loss(pred_dx, target_dx)
    )
    total = l1_weight * l1 + cosine_weight * cosine + gradient_weight * gradient
    return total, {
        "loss": total.detach(),
        "l1": l1.detach(),
        "cosine": cosine.detach(),
        "gradient": gradient.detach(),
        "valid_fraction": valid.float().mean().detach(),
    }
