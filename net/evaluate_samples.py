import argparse
import csv
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torchvision.transforms as tfs
from PIL import Image

from net.data_utils import VALID_EXTENSIONS, image_paths
from net.metrics import psnr, ssim
from net.runtime import resolve_path

VALID_EXTS = VALID_EXTENSIONS


def image_id(pred_path):
    name = pred_path.stem
    if name.endswith("_FFA"):
        name = name[:-4]
    return name.split("_")[0]


def load_tensor(path):
    with Image.open(path) as source:
        image = source.convert("RGB")
    return tfs.ToTensor()(image).unsqueeze(0)


def find_gt(gt_dir, sample_id):
    for ext in sorted(VALID_EXTS):
        candidate = gt_dir / f"{sample_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def evaluate(pred_dir, gt_dir, limit=None):
    pred_paths = image_paths(pred_dir)
    if limit is not None:
        pred_paths = pred_paths[:limit]

    rows = []
    missing = []
    for pred_path in pred_paths:
        sample_id = image_id(pred_path)
        gt_path = find_gt(gt_dir, sample_id)
        if gt_path is None:
            missing.append(pred_path.name)
            continue

        pred = load_tensor(pred_path)
        gt = load_tensor(gt_path)
        if pred.shape != gt.shape:
            raise ValueError(
                f"shape mismatch for {pred_path.name}: "
                f"pred {tuple(pred.shape)} vs gt {tuple(gt.shape)}"
            )

        with torch.no_grad():
            sample_psnr = psnr(pred, gt)
            sample_ssim = ssim(pred, gt).item()
        rows.append({
            "image": pred_path.name,
            "gt": gt_path.name,
            "psnr": sample_psnr,
            "ssim": sample_ssim,
        })

    return rows, missing


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "gt", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path, rows, mean_psnr, mean_ssim):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    indices = list(range(1, len(rows) + 1))
    labels = [Path(row["image"]).stem.replace("_FFA", "") for row in rows]
    psnrs = [row["psnr"] for row in rows]
    ssims = [row["ssim"] for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#f7f7f2")

    psnr_color = "#2f6f73"
    ssim_color = "#b15d3a"
    grid_color = "#d8d3c8"
    text_color = "#242424"

    axes[0].plot(indices, psnrs, color=psnr_color, linewidth=1.8)
    axes[0].scatter(indices, psnrs, color=psnr_color, s=18, alpha=0.85)
    axes[0].axhline(mean_psnr, color="#1f3f42", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title(
        f"Haze4K sample evaluation | mean PSNR {mean_psnr:.4f} dB",
        loc="left",
        color=text_color,
        fontsize=13,
        fontweight="bold",
    )

    axes[1].plot(indices, ssims, color=ssim_color, linewidth=1.8)
    axes[1].scatter(indices, ssims, color=ssim_color, s=18, alpha=0.85)
    axes[1].axhline(mean_ssim, color="#76351f", linestyle="--", linewidth=1.2)
    axes[1].set_ylabel("SSIM")
    axes[1].set_title(
        f"Structural similarity | mean SSIM {mean_ssim:.6f}",
        loc="left",
        color=text_color,
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Sample")

    tick_step = max(1, len(indices) // 20)
    tick_indices = indices[::tick_step]
    tick_labels = labels[::tick_step]
    axes[1].set_xticks(tick_indices)
    axes[1].set_xticklabels(tick_labels, rotation=45, ha="right")

    for axis in axes:
        axis.set_facecolor("#fbfaf7")
        axis.grid(True, color=grid_color, linewidth=0.8, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#99968e")
        axis.spines["bottom"].set_color("#99968e")
        axis.tick_params(colors=text_color)

    fig.tight_layout(pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate dehazed sample images with PSNR and SSIM."
    )
    parser.add_argument(
        "--pred_dir",
        default="samples/haze4k_test_check",
        help="Folder containing predicted images.",
    )
    parser.add_argument(
        "--gt_dir",
        default="data/Haze4K/test/gt",
        help="Folder containing ground-truth images.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save per-image metrics as CSV.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional path to save a PNG chart of PSNR and SSIM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N images.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print the summary.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    pred_dir = resolve_path(args.pred_dir)
    gt_dir = resolve_path(args.gt_dir)
    if not pred_dir.exists():
        raise FileNotFoundError(f"cannot find pred_dir: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"cannot find gt_dir: {gt_dir}")

    rows, missing = evaluate(pred_dir, gt_dir, args.limit)
    if not rows:
        raise RuntimeError(f"no matched images found in {pred_dir}")

    if not args.quiet:
        print("image,gt,psnr,ssim")
        for row in rows:
            print(
                f"{row['image']},{row['gt']},"
                f"{row['psnr']:.4f},{row['ssim']:.6f}"
            )

    mean_psnr = sum(row["psnr"] for row in rows) / len(rows)
    mean_ssim = sum(row["ssim"] for row in rows) / len(rows)
    print(f"evaluated: {len(rows)}")
    print(f"mean PSNR: {mean_psnr:.4f}")
    print(f"mean SSIM: {mean_ssim:.6f}")

    if missing:
        print(f"missing gt: {len(missing)}")
        print("first missing:", ", ".join(missing[:10]))

    if args.csv:
        csv_path = resolve_path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, rows)
        print(f"csv saved: {csv_path}")

    if args.plot:
        plot_path = resolve_path(args.plot)
        write_plot(plot_path, rows, mean_psnr, mean_ssim)
        print(f"plot saved: {plot_path}")


if __name__ == "__main__":
    main()
