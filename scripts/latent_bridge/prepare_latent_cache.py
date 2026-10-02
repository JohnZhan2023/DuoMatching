#!/usr/bin/env python3
"""Cache Wan latent pairs and frame-specific Qwen-Image targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from latent_bridge.config import load_yaml, require


LATENT_MEAN = (
    -0.7571,
    -0.7089,
    -0.9113,
    0.1075,
    -0.1745,
    0.9653,
    -0.1517,
    1.5508,
    0.4134,
    -0.0715,
    0.5517,
    -0.3632,
    -0.1922,
    -0.9497,
    0.2503,
    -0.2921,
)
LATENT_STD = (
    2.8184,
    1.4541,
    2.3275,
    2.6558,
    1.2196,
    1.7708,
    2.6052,
    2.0743,
    3.2687,
    2.1526,
    2.8652,
    1.5579,
    1.6382,
    1.1253,
    2.8251,
    1.9160,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Process at most this many manifest records per rank (for smoke tests).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cache files that already exist.",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, str):
            item = {"video": item}
        if not isinstance(item, dict) or "video" not in item:
            raise ValueError(f"Invalid manifest record at {path}:{line_number}")
        records.append(item)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def resize_center_crop(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    scale = max(height / source_h, width / source_w)
    resized_h = max(height, round(source_h * scale))
    resized_w = max(width, round(source_w * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    top = (resized_h - height) // 2
    left = (resized_w - width) // 2
    return np.ascontiguousarray(resized[top : top + height, left : left + width])


def read_video(path: Path, height: int, width: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(resize_center_crop(frame, height, width))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {path}")
    return frames


def choose_clip_starts(
    frame_count: int,
    frames_per_clip: int,
    clips_per_video: int,
    seed: int,
) -> list[int]:
    last_start = frame_count - frames_per_clip
    if last_start < 0:
        return []
    if clips_per_video == 1:
        return [random.Random(seed).randint(0, last_start)]
    candidates = np.linspace(0, last_start, num=clips_per_video, dtype=np.int64)
    return sorted(set(int(value) for value in candidates))


def frames_to_tensor(frames: Iterable[np.ndarray], device: torch.device) -> torch.Tensor:
    rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames])
    tensor = torch.from_numpy(rgb).to(device=device, dtype=torch.float32)
    return tensor.permute(0, 3, 1, 2).div_(127.5).sub_(1.0)


def load_wan_vae(
    repo: Path,
    checkpoint: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from wan.modules.vae import _video_vae

    return (
        _video_vae(pretrained_path=str(checkpoint), z_dim=16)
        .eval()
        .requires_grad_(False)
        .to(device=device, dtype=dtype)
    )


def load_qwen_vae(
    model_root: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    from diffusers import AutoencoderKLQwenImage

    model_path = model_root / "vae" if (model_root / "vae").is_dir() else model_root
    return (
        AutoencoderKLQwenImage.from_pretrained(str(model_path), torch_dtype=dtype)
        .eval()
        .requires_grad_(False)
        .to(device)
    )


@torch.inference_mode()
def encode_wan_clip(
    vae,
    frames: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    # [T,3,H,W] -> [1,3,T,H,W] -> [F,16,h,w].
    video = frames.permute(1, 0, 2, 3).unsqueeze(0).to(dtype=dtype)
    mean = torch.tensor(LATENT_MEAN, device=video.device, dtype=dtype)
    inv_std = torch.tensor(LATENT_STD, device=video.device, dtype=dtype).reciprocal()
    latent = vae.encode(video, [mean, inv_std])
    return latent[0].permute(1, 0, 2, 3).float()


@torch.inference_mode()
def encode_qwen_frames(
    vae,
    frames: torch.Tensor,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    """Encode each RGB frame independently into normalized Qwen image latents."""
    outputs = []
    for batch in frames.split(batch_size):
        images = batch.to(dtype=dtype)
        latent = vae.encode(images.unsqueeze(2)).latent_dist.mode()
        mean = torch.as_tensor(
            vae.config.latents_mean, device=frames.device, dtype=dtype
        ).view(1, vae.config.z_dim, 1, 1, 1)
        inv_std = torch.as_tensor(
            vae.config.latents_std, device=frames.device, dtype=dtype
        ).reciprocal().view(1, vae.config.z_dim, 1, 1, 1)
        outputs.append(((latent - mean) * inv_std)[:, :, 0].float())
    return torch.cat(outputs)


def storage_dtype(name: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return choices[name]
    except KeyError as error:
        raise ValueError(f"Unsupported storage_dtype: {name}") from error


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    data_cfg = require(config, "data")
    model_cfg = require(config, "models")
    cache_cfg = require(config, "cache")
    if str(model_cfg.get("target_type", "qwen")).lower() != "qwen":
        raise ValueError("LatentBridge preprocessing supports Qwen-Image only")

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Ampere and newer have native BF16 support. V100-class GPUs need FP16.
        compute_dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(device)[0] >= 8
            else torch.float16
        )
    else:
        compute_dtype = torch.float32

    manifest = Path(require(data_cfg, "manifest"))
    records = read_manifest(manifest)[rank::world_size]
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")
        records = records[: args.max_records]
    output_dir = Path(require(cache_cfg, "output_dir"))
    output_dir.mkdir(parents=True, exist_ok=True)

    height = int(data_cfg.get("height", 480))
    width = int(data_cfg.get("width", 832))
    chunks_per_clip = int(data_cfg.get("chunks_per_clip", 4))
    clips_per_video = int(data_cfg.get("clips_per_video", 1))
    seed = int(data_cfg.get("seed", 0))
    target_batch_size = int(
        cache_cfg.get("target_batch_size", cache_cfg.get("qwen_batch_size", 4))
    )
    save_dtype = storage_dtype(cache_cfg.get("storage_dtype", "float16"))
    frames_per_clip = 1 + 4 * chunks_per_clip

    wan_vae = load_wan_vae(
        Path(require(model_cfg, "causal_forcing_repo")),
        Path(require(model_cfg, "wan_vae")),
        device,
        compute_dtype,
    )
    legacy_target = model_cfg.get("qwen_image")
    target_model = model_cfg.get("target_model", legacy_target)
    if target_model is None:
        raise KeyError("models.target_model is required")
    target_vae = load_qwen_vae(
        Path(target_model),
        device,
        compute_dtype,
    )

    written = 0
    skipped_short = 0
    failures = []
    for record in tqdm(records, disable=rank != 0):
        video_path = Path(record["video"])
        try:
            video_frames = read_video(video_path, height, width)
            identity = str(record.get("id", video_path.resolve()))
            record_seed = seed + int(
                hashlib.sha1(identity.encode()).hexdigest()[:8],
                16,
            )
            starts = choose_clip_starts(
                len(video_frames),
                frames_per_clip,
                clips_per_video,
                record_seed,
            )
            if not starts:
                skipped_short += 1
                continue
            for start in starts:
                clip_frames = video_frames[start : start + frames_per_clip]
                pixels = frames_to_tensor(clip_frames, device)
                wan_latents = encode_wan_clip(wan_vae, pixels, compute_dtype)
                expected_latents = 1 + chunks_per_clip
                if wan_latents.shape[0] != expected_latents:
                    raise RuntimeError(
                        f"Expected {expected_latents} Wan latents, "
                        f"got {tuple(wan_latents.shape)}"
                    )
                target_latents = encode_qwen_frames(
                    target_vae,
                    pixels[1:],
                    compute_dtype,
                    target_batch_size,
                )
                target_latents = target_latents.reshape(
                    chunks_per_clip, 4, *target_latents.shape[1:]
                )

                digest = hashlib.sha1(
                    f"{identity}:{start}:{height}:{width}".encode()
                ).hexdigest()[:16]
                for chunk_index in range(chunks_per_clip):
                    sample_id = f"{digest}_c{chunk_index:03d}"
                    output_path = output_dir / f"sample_{sample_id}.pt"
                    if output_path.exists() and not args.overwrite:
                        continue
                    payload = {
                        "previous_wan_latent": wan_latents[chunk_index]
                        .to(device="cpu", dtype=save_dtype)
                        .contiguous(),
                        "current_wan_latent": wan_latents[chunk_index + 1]
                        .to(device="cpu", dtype=save_dtype)
                        .contiguous(),
                        "target_latents": target_latents[chunk_index]
                        .to(device="cpu", dtype=save_dtype)
                        .contiguous(),
                        "target_type": "qwen",
                        "sample_id": sample_id,
                        "video": str(video_path),
                        "clip_start": start,
                        "chunk_index": chunk_index,
                        "target_frame_indices": [
                            start + 1 + 4 * chunk_index + offset
                            for offset in range(4)
                        ],
                    }
                    torch.save(payload, output_path)
                    written += 1
        except Exception as error:
            failures.append({"video": str(video_path), "error": repr(error)})

    metadata = {
        "rank": rank,
        "world_size": world_size,
        "written": written,
        "skipped_short": skipped_short,
        "failures": failures,
        "config": str(args.config.resolve()),
    }
    (output_dir / f"prepare_rank_{rank:04d}.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        print(f"Rank {rank}: {len(failures)} videos failed; see metadata JSON")
    print(f"Rank {rank}: wrote {written} samples to {output_dir}")


if __name__ == "__main__":
    main()
