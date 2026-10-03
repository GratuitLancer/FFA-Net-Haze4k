"""Inference entry point, preserving original task and checkpoint conventions."""

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image

from net.data_utils import image_paths, normalize_haze, tensorShow
from net.models import FFA
from net.runtime import PROJECT_ROOT, find_checkpoint, load_checkpoint, resolve_device, resolve_path


def build_parser():
    parser = argparse.ArgumentParser(description="Dehaze images with FFA-Net.")
    parser.add_argument("--task", default="haze4k", choices=("its", "ots", "haze4k"))
    parser.add_argument("--test_imgs", default="test_imgs", help="Input folder")
    parser.add_argument("--model_dir", default=None, help="Checkpoint file; auto-detected when omitted")
    parser.add_argument("--output_dir", default=None, help="Defaults to pred_FFA_<task>")
    parser.add_argument("--gps", type=int, choices=(3,), default=3)
    parser.add_argument("--blocks", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    return parser


def resolve_input_path(path):
    source = Path(path).expanduser()
    root_path = resolve_path(source)
    if source.is_absolute() or root_path.exists():
        return root_path
    legacy_path = resolve_path(source, PROJECT_ROOT / "net")
    return legacy_path if legacy_path.exists() else root_path


def main(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    if options.limit is not None and options.limit <= 0:
        parser.error("--limit must be positive")
    blocks = options.blocks if options.blocks is not None else (20 if options.task == "haze4k" else 19)
    if blocks <= 0:
        parser.error("--blocks must be positive")
    device = resolve_device(options.device)
    img_dir = resolve_input_path(options.test_imgs)
    model_path = resolve_input_path(options.model_dir) if options.model_dir else find_checkpoint(options.task, options.gps, blocks)
    if not model_path.is_file():
        raise FileNotFoundError(f"Cannot find model: {model_path}")
    paths = image_paths(img_dir)
    if options.limit is not None:
        paths = paths[:options.limit]
    if not paths:
        raise ValueError(f"No input images found in {img_dir}")
    model = FFA(gps=options.gps, blocks=blocks).to(device)
    load_checkpoint(model_path, model, device)
    model.eval()
    output_dir = resolve_path(options.output_dir or f"pred_FFA_{options.task}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print("model_dir:", model_path)
    print("pred_dir:", output_dir)
    with torch.inference_mode():
        for path in paths:
            with Image.open(path) as source:
                haze = source.convert("RGB")
            prediction = model(normalize_haze(haze).unsqueeze(0).to(device)).clamp(0, 1).cpu()
            if options.show:
                tensorShow([to_tensor(haze).unsqueeze(0), prediction], ["haze", "pred"])
            destination = output_dir / f"{path.stem}_FFA.png"
            save_image(prediction.squeeze(0), destination)
            print(destination.name)
    return output_dir


if __name__ == "__main__":
    main()
