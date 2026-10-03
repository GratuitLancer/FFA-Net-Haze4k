"""Training CLI configuration. Importing this module has no side effects."""

import argparse

from net.runtime import PROJECT_ROOT, resolve_device, resolve_path


DATASETS = ("its_train", "its_test", "ots_train", "ots_test", "haze4k_train", "haze4k_test")


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def build_parser():
    parser = argparse.ArgumentParser(description="Train the original FFA-Net model.")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--resume", type=parse_bool, nargs="?", const=True, default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--eval_step", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--model_dir", default="trained_models", help="Checkpoint directory or file")
    parser.add_argument("--data_root", default="data", help="Dataset directory, relative to the project root")
    parser.add_argument("--trainset", choices=DATASETS[::2], default="haze4k_train")
    parser.add_argument("--testset", choices=DATASETS[1::2], default="haze4k_test")
    parser.add_argument("--net", choices=("ffa",), default="ffa")
    parser.add_argument("--gps", type=int, choices=(3,), default=3, help="Residual groups")
    parser.add_argument("--blocks", type=int, default=20, help="Residual blocks per group")
    parser.add_argument("--bs", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop", dest="crop", action="store_true", default=True)
    parser.add_argument("--no_crop", dest="crop", action="store_false")
    parser.add_argument("--crop_size", type=int, default=240)
    parser.add_argument("--no_lr_sche", action="store_true")
    parser.add_argument("--perloss", action="store_true", help="Enable VGG perceptual loss")
    return parser


def parse_options(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    for name in ("steps", "eval_step", "blocks", "bs", "crop_size"):
        if getattr(options, name) <= 0:
            parser.error(f"--{name} must be positive")
    if options.lr <= 0 or options.num_workers < 0:
        parser.error("--lr must be positive and --num_workers must be non-negative")
    options.device = resolve_device(options.device)
    options.data_root = resolve_path(options.data_root)
    options.model_name = f"{options.trainset}_{options.net}_{options.gps}_{options.blocks}"
    model_path = resolve_path(options.model_dir)
    if model_path.suffix.lower() in {".pk", ".pt", ".pth", ".ckpt"}:
        options.model_dir = model_path
    else:
        options.model_dir = model_path / f"{options.model_name}.pk"
    options.log_dir = PROJECT_ROOT / "logs" / options.model_name
    options.history_dir = PROJECT_ROOT / "numpy_files"
    return options
