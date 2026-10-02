#!/usr/bin/env python3
"""Train the frame-index-conditioned Wan-to-Qwen latent adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from latent_bridge.config import load_yaml, require
from latent_bridge.data import CachedLatentDataset, split_cached_datasets
from latent_bridge.losses import adapter_loss
from latent_bridge.model import FrameLatentAdapter, FrameLatentAdapterConfig
from latent_bridge.schedule import cosine_schedule, distributed_schedule_steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Accelerate state directory written by a previous checkpoint.",
    )
    return parser.parse_args()


def step_from_resume_path(path: Path | None) -> int:
    if path is None:
        return 0
    for candidate in (path, path.parent):
        if candidate.name.startswith("checkpoint-"):
            try:
                return int(candidate.name.removeprefix("checkpoint-"))
            except ValueError:
                pass
    raise ValueError(
        "--resume must point to a training_state directory inside "
        "checkpoint-<step>, so the global step can be recovered"
    )


def load_compatible_initialization(
    model: FrameLatentAdapter,
    adapter_dir: str | Path,
) -> tuple[int, list[str]]:
    """Warm-start parameters whose names and shapes match another adapter."""
    adapter_dir = Path(adapter_dir)
    source = torch.load(
        adapter_dir / FrameLatentAdapter.weights_name,
        map_location="cpu",
        weights_only=True,
    )
    destination = model.state_dict()
    compatible = {
        name: value
        for name, value in source.items()
        if name in destination and value.shape == destination[name].shape
    }
    model.load_state_dict(compatible, strict=False)
    skipped = sorted(set(source).difference(compatible))
    return len(compatible), skipped


@torch.no_grad()
def validate(
    model: FrameLatentAdapter,
    loader: DataLoader,
    accelerator: Accelerator,
    loss_cfg: dict,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(4, device=accelerator.device, dtype=torch.float64)
    count = torch.zeros(1, device=accelerator.device, dtype=torch.float64)
    for batch in loader:
        with accelerator.autocast():
            prediction = model(
                batch["previous_latent"],
                batch["current_latent"],
                batch["frame_offset"],
            )
            _, metrics = adapter_loss(
                prediction,
                batch["target_latent"],
                **loss_cfg,
            )
        batch_size = prediction.shape[0]
        totals += batch_size * torch.stack(
            [
                metrics["loss"],
                metrics["l1"],
                metrics["cosine"],
                metrics["gradient"],
            ]
        ).double()
        count += batch_size
    totals = accelerator.reduce(totals, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    values = (totals / count.clamp_min(1)).tolist()
    model.train()
    return dict(zip(("val/loss", "val/l1", "val/cosine", "val/gradient"), values))


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    training_cfg = require(config, "training")
    data_cfg = require(config, "data")
    loss_cfg = require(config, "loss")

    output_dir = Path(require(training_cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)
    use_wandb = bool(training_cfg.get("wandb", False))
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            training_cfg.get("gradient_accumulation_steps", 1)
        ),
        mixed_precision=str(training_cfg.get("mixed_precision", "bf16")),
        log_with="wandb" if use_wandb else None,
        project_dir=str(output_dir),
    )
    if use_wandb:
        accelerator.init_trackers(
            project_name=str(training_cfg.get("wandb_project", "latent-bridge")),
            config=config,
        )

    seed = int(training_cfg.get("seed", 0))
    torch.manual_seed(seed + accelerator.process_index)
    cache_dir = require(data_cfg, "cache_dir")
    expand_frame_offsets = bool(data_cfg.get("expand_frame_offsets", True))
    validation_cache_dir = data_cfg.get("validation_cache_dir")
    if validation_cache_dir is not None:
        train_dataset = CachedLatentDataset(
            cache_dir, expand_frame_offsets=expand_frame_offsets
        )
        validation_dataset = CachedLatentDataset(
            validation_cache_dir,
            expand_frame_offsets=expand_frame_offsets,
        )
    else:
        validation_fraction = float(data_cfg.get("validation_fraction", 0.02))
        train_dataset, validation_dataset = split_cached_datasets(
            cache_dir,
            validation_fraction=validation_fraction,
            seed=seed,
            expand_frame_offsets=expand_frame_offsets,
        )
    batch_size = int(training_cfg.get("batch_size", 8))
    if len(train_dataset) < batch_size:
        raise ValueError(
            f"Training split has {len(train_dataset)} items, fewer than "
            f"batch_size={batch_size}"
        )
    num_workers = int(data_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_cfg.get("validation_batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    model = FrameLatentAdapter(
        FrameLatentAdapterConfig(**require(config, "adapter"))
    )
    init_adapter = training_cfg.get("init_adapter")
    if init_adapter is not None and args.resume is None:
        loaded_count, skipped = load_compatible_initialization(model, init_adapter)
        if accelerator.is_main_process:
            print(
                f"Warm-started {loaded_count} tensors from {init_adapter}; "
                f"shape-incompatible tensors: {skipped}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-4)),
        betas=tuple(training_cfg.get("betas", [0.9, 0.95])),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )
    max_steps = int(training_cfg.get("max_steps", 100_000))
    # Accelerate's default AcceleratedScheduler advances once per process when
    # batches are sharded. Express the schedule in optimizer-step units so a
    # four-process run does not exhaust a 100k schedule at global step 25k.
    scheduler_warmup_steps, scheduler_total_steps = distributed_schedule_steps(
        int(training_cfg.get("warmup_steps", 1_000)),
        max_steps,
        accelerator.num_processes,
    )
    scheduler = cosine_schedule(
        optimizer,
        scheduler_warmup_steps,
        scheduler_total_steps,
    )
    model, optimizer, train_loader, validation_loader, scheduler = accelerator.prepare(
        model,
        optimizer,
        train_loader,
        validation_loader,
        scheduler,
    )
    if args.resume is not None:
        accelerator.load_state(str(args.resume))

    loss_kwargs = {
        "l1_weight": float(loss_cfg.get("l1_weight", 1.0)),
        "cosine_weight": float(loss_cfg.get("cosine_weight", 0.1)),
        "gradient_weight": float(loss_cfg.get("gradient_weight", 0.1)),
    }
    log_every = int(training_cfg.get("log_every", 20))
    validate_every = int(training_cfg.get("validate_every", 1_000))
    save_every = int(training_cfg.get("save_every", 5_000))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

    global_step = step_from_resume_path(args.resume)
    model.train()
    while global_step < max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                with accelerator.autocast():
                    prediction = model(
                        batch["previous_latent"],
                        batch["current_latent"],
                        batch["frame_offset"],
                    )
                    loss, metrics = adapter_loss(
                        prediction,
                        batch["target_latent"],
                        **loss_kwargs,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1

            if global_step % log_every == 0:
                log_values = {
                    f"train/{key}": value.item() for key, value in metrics.items()
                }
                log_values["train/lr"] = scheduler.get_last_lr()[0]
                accelerator.log(log_values, step=global_step)
                if accelerator.is_main_process:
                    print(
                        f"step={global_step} loss={log_values['train/loss']:.6f} "
                        f"cos={log_values['train/cosine']:.6f} "
                        f"lr={log_values['train/lr']:.8g}"
                    )

            if global_step % validate_every == 0:
                validation_metrics = validate(
                    model,
                    validation_loader,
                    accelerator,
                    loss_kwargs,
                )
                accelerator.log(validation_metrics, step=global_step)
                if accelerator.is_main_process:
                    print(
                        f"validation step={global_step} "
                        f"loss={validation_metrics['val/loss']:.6f}"
                    )

            if global_step % save_every == 0:
                checkpoint_dir = output_dir / f"checkpoint-{global_step:08d}"
                accelerator.save_state(str(checkpoint_dir / "training_state"))
                if accelerator.is_main_process:
                    accelerator.unwrap_model(model).save_pretrained(
                        checkpoint_dir / "adapter"
                    )
                    (checkpoint_dir / "source_config.yaml").write_text(
                        args.config.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                accelerator.wait_for_everyone()

            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(output_dir / "final")
        (output_dir / "source_config.yaml").write_text(
            args.config.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
