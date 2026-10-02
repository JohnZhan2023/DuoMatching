from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = {
    "previous_wan_latent",
    "current_wan_latent",
}


def _read_cache_index(cache_dir: Path) -> list[Path]:
    index_path = cache_dir / "index.jsonl"
    if index_path.is_file():
        paths = []
        for line_number, line in enumerate(
            index_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            item = json.loads(line)
            relative = item["file"] if isinstance(item, dict) else item
            path = cache_dir / relative
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing cache file from index line {line_number}: {path}"
                )
            paths.append(path)
        return paths
    return sorted(cache_dir.glob("sample_*.pt"))


class CachedLatentDataset(Dataset[dict[str, Any]]):
    """Dataset of cached Wan latent pairs and four Qwen frame-latent targets."""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        expand_frame_offsets: bool = True,
        load_dtype: torch.dtype = torch.float32,
        files: list[Path] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.files = list(files) if files is not None else _read_cache_index(self.cache_dir)
        if not self.files:
            raise FileNotFoundError(f"No cached .pt samples found in {self.cache_dir}")
        self.expand_frame_offsets = expand_frame_offsets
        self.load_dtype = load_dtype

    def __len__(self) -> int:
        multiplier = 4 if self.expand_frame_offsets else 1
        return len(self.files) * multiplier

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.expand_frame_offsets:
            file_index, frame_offset = divmod(index, 4)
        else:
            file_index = index
            frame_offset = int(torch.randint(0, 4, ()).item())

        payload = torch.load(
            self.files[file_index],
            map_location="cpu",
            weights_only=True,
        )
        missing = REQUIRED_KEYS.difference(payload)
        if missing:
            raise KeyError(f"{self.files[file_index]} is missing keys: {sorted(missing)}")

        if payload.get("target_type", "qwen") != "qwen":
            raise ValueError("LatentBridge cache must contain Qwen-Image targets")
        previous = payload["previous_wan_latent"]
        current = payload["current_wan_latent"]
        target_key = "target_latents"
        if target_key not in payload:
            # Backward compatibility with the original Qwen-only cache.
            target_key = "target_qwen_latents"
        if target_key not in payload:
            raise KeyError(f"{self.files[file_index]} has no target latents")
        targets = payload[target_key]
        if previous.ndim != 3 or current.shape != previous.shape:
            raise ValueError(f"Invalid Wan latent shapes in {self.files[file_index]}")
        if targets.shape != (4, *previous.shape):
            raise ValueError(
                "Expected four Qwen targets matching the Wan latent shape, "
                f"got {tuple(targets.shape)} in {self.files[file_index]}"
            )

        return {
            "previous_latent": previous.to(self.load_dtype),
            "current_latent": current.to(self.load_dtype),
            "target_latent": targets[frame_offset].to(self.load_dtype),
            "frame_offset": torch.tensor(frame_offset, dtype=torch.long),
            "sample_id": payload.get("sample_id", self.files[file_index].stem),
        }


def split_cached_datasets(
    cache_dir: str | Path,
    *,
    validation_fraction: float,
    seed: int,
    expand_frame_offsets: bool = True,
    load_dtype: torch.dtype = torch.float32,
) -> tuple[CachedLatentDataset, CachedLatentDataset]:
    """Split whole cache records so their four offsets cannot leak across sets."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    cache_dir = Path(cache_dir)
    files = _read_cache_index(cache_dir)
    if len(files) < 2:
        raise ValueError("at least two cache files are required for a split")
    validation_size = max(1, round(len(files) * validation_fraction))
    if validation_size >= len(files):
        raise ValueError("validation split leaves no training cache files")
    order = torch.randperm(
        len(files), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    validation_indices = set(order[:validation_size])
    train_files = [path for index, path in enumerate(files) if index not in validation_indices]
    validation_files = [path for index, path in enumerate(files) if index in validation_indices]
    kwargs = {
        "expand_frame_offsets": expand_frame_offsets,
        "load_dtype": load_dtype,
    }
    return (
        CachedLatentDataset(cache_dir, files=train_files, **kwargs),
        CachedLatentDataset(cache_dir, files=validation_files, **kwargs),
    )
