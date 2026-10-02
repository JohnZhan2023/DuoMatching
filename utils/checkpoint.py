"""Strict generator loading for released and FSDP training checkpoints."""
import torch


def load_generator_checkpoint(generator, path, key="generator_ema"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if key not in checkpoint:
        raise KeyError(f"Checkpoint has {list(checkpoint)}, expected {key!r}; "
                       "released DuoMatching weights require --use_ema")
    state = {}
    for name, value in checkpoint[key].items():
        name = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        if name in state:
            raise ValueError(f"Duplicate checkpoint key after normalization: {name}")
        state[name] = value
    generator.load_state_dict(state, strict=True)
