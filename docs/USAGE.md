# Usage

This page contains the operational details for training, resuming, inference, and evaluation.

## Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For NVIDIA GPU training, install the CUDA-enabled PyTorch build matching your local driver and CUDA environment.

Verify CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Haze4K dataset

Expected layout:

```text
data/Haze4K/
├── train/
│   ├── haze/
│   ├── gt/
│   └── trans/
└── test/
    ├── haze/
    ├── gt/
    └── trans/
```

The current pipeline uses `haze/` and `gt/`.

A hazy filename such as:

```text
1000_0.74_1.6.png
```

is paired with:

```text
1000.png
```

The identifier before the first underscore is used as the pairing key.

## Train FFA-Net on Haze4K

Recommended module entry point:

```bash
python -m net.main \
  --data_root data \
  --trainset haze4k_train \
  --testset haze4k_test \
  --blocks 20 \
  --gps 3 \
  --bs 2 \
  --crop_size 240 \
  --lr 0.0001 \
  --steps 100000 \
  --eval_step 5000
```

The legacy script entry point also remains supported:

```bash
python net/main.py --data_root data --trainset haze4k_train --testset haze4k_test
```

CUDA is selected automatically when available unless `--device` is supplied.

Examples:

```bash
python -m net.main --device cpu
python -m net.main --device cuda:0
```

## Cropping and paired augmentation

Haze4K images have different spatial dimensions, so training normally uses random crops.

The loader applies the same crop, flip, and rotation to both the hazy input and ground truth.

If either image is smaller than the requested crop size, the pair is resized together first. This avoids the unbounded re-sampling behavior that can occur when repeatedly selecting undersized images.

## Checkpoints

The best model keeps the original FFA-Net-style filename:

```text
trained_models/haze4k_train_ffa_3_20.pk
```

A companion latest-progress checkpoint is written as:

```text
trained_models/haze4k_train_ffa_3_20_last.pk
```

The latest checkpoint includes:

- model parameters
- optimizer state
- current step
- loss history
- PSNR history
- SSIM history
- best PSNR / SSIM values

Resume prefers `*_last.pk` and falls back to the best-model checkpoint.

Legacy checkpoints without optimizer state remain loadable.

To train from scratch:

```bash
python -m net.main --no_resume
```

or:

```bash
python -m net.main --resume False
```

## Best-model rule

The original save rule is preserved:

> a new best checkpoint is written only when both PSNR and SSIM improve.

This is intentionally retained for compatibility with the original workflow.

## Inference

Run the full Haze4K test split:

```bash
python -m net.test \
  --task haze4k \
  --test_imgs data/Haze4K/test/haze \
  --output_dir samples/haze4k_predictions
```

Quick single-image check:

```bash
python -m net.test \
  --task haze4k \
  --test_imgs data/Haze4K/test/haze \
  --output_dir samples/haze4k_check \
  --limit 1
```

Use a custom checkpoint:

```bash
python -m net.test \
  --task haze4k \
  --model_dir net/trained_models/haze4k_train_ffa_3_20.pk \
  --test_imgs data/Haze4K/test/haze
```

Predictions are saved with the `_FFA.png` suffix.

## Evaluate saved predictions

```bash
python -m net.evaluate_samples \
  --pred_dir samples/haze4k_predictions \
  --gt_dir data/Haze4K/test/gt \
  --csv samples/metrics.csv \
  --plot samples/metrics.png
```

The evaluator:

1. matches predictions to ground truth,
2. checks image shapes,
3. computes PSNR,
4. computes SSIM,
5. prints the mean,
6. optionally writes per-image CSV output,
7. optionally writes a plot.

Evaluate only the first N outputs:

```bash
python -m net.evaluate_samples \
  --pred_dir samples/haze4k_predictions \
  --gt_dir data/Haze4K/test/gt \
  --limit 20
```

## Tests

Run the CPU-only regression suite:

```bash
python -m unittest discover -s tests -v
```

The tests do not require downloading Haze4K or pretrained checkpoints.

They cover:

- paired augmentation
- Haze4K pairing
- RESIDE pairing
- import side effects
- argument parsing
- legacy checkpoint loading
- DataParallel `module.` prefix handling
- optimizer restoration
- consecutive batch traversal
- saved-image evaluation

## Main modules

| Module | Responsibility |
|---|---|
| `net/main.py` | training entry point |
| `net/option.py` | CLI configuration |
| `net/data_utils.py` | dataset pairing, crop, augmentation, loaders |
| `net/engine.py` | training, validation, scheduling, resume |
| `net/runtime.py` | paths, devices, checkpoint compatibility |
| `net/test.py` | inference |
| `net/evaluate_samples.py` | saved-image metrics and exports |
| `net/metrics.py` | PSNR / SSIM |
| `net/models/FFA.py` | original FFA-Net architecture |
