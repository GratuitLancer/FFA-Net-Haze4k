"""Shared project paths, device selection and checkpoint compatibility."""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(path, base=PROJECT_ROOT):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_device(device="auto"):
    if device in {"auto", "Automatic detection"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("FFA-Net supports cpu and cuda devices")
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if selected.index is not None and selected.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {selected.index} does not exist")
    return str(selected)


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_checkpoint(path, model, device="cpu", optimizer=None):
    # Original training checkpoints contain histories and NumPy scalar metadata.
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    unwrap_model(model).load_state_dict(state)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def save_checkpoint(path, model, optimizer, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(metadata, model=unwrap_model(model).state_dict(), optimizer=optimizer.state_dict())
    torch.save(checkpoint, path)


def find_checkpoint(task, gps, blocks):
    name = f"{task}_train_ffa_{gps}_{blocks}.pk"
    candidates = (PROJECT_ROOT / "trained_models" / name, PROJECT_ROOT / "net" / "trained_models" / name)
    return next((path for path in candidates if path.is_file()), candidates[0])
