# FFA-Net-Haze4k for Single Image Dehazing

Fork: <https://github.com/GratuitLancer/FFA-Net-Haze4k>

This fork retains the original PyTorch FFA model, attention blocks, residual
groups, loss functions, and PSNR/SSIM metrics. The workflow is lightly refactored
to separate configuration, dataset construction, training, and checkpoint I/O.
Existing Haze4K support and original checkpoint parameter names are preserved.

## COSC428 Research Project

This repository contains an adaptation of **FFA-Net (Feature Fusion Attention
Network)** for a COSC428 Research Project on single-image dehazing. The project
uses the Haze4K dataset to train and evaluate a deep neural network that
reconstructs clear images from hazy observations.

The implementation is based on the original FFA-Net work by Qin et al. and has
been modified to support:

- Haze4K training and test splits;
- automatic hazy/ground-truth image pairing;
- GPU training with CUDA-enabled PyTorch;
- Haze4K checkpoint discovery and inference;
- configurable training, evaluation, and prediction paths.

## Research Objective

Image dehazing aims to recover scene visibility, colour, and structural detail
that have been degraded by atmospheric haze. This project investigates how the
channel-attention and pixel-attention mechanisms in FFA-Net perform when trained
on Haze4K.

The main experimental workflow is:

1. Train FFA-Net using paired hazy and ground-truth images.
2. Evaluate the model on the Haze4K test split.
3. Generate dehazed predictions for qualitative inspection.
4. Compare reconstructed images using PSNR and SSIM.

## Haze4K Dataset

The Haze4K dataset can be downloaded from Baidu Netdisk:

- Download link: <https://pan.baidu.com/share/init?surl=41MW0YAvjFcydlroQZZizA>
- Password: `cmmr`

After downloading, extract the dataset into `data/Haze4K` using the following
structure:

```text
FFA-Net/
|-- data/
|   `-- Haze4K/
|       |-- train/
|       |   |-- haze/
|       |   |   `-- *.png
|       |   |-- gt/
|       |   |   `-- *.png
|       |   `-- trans/
|       `-- test/
|           |-- haze/
|           |   `-- *.png
|           |-- gt/
|           |   `-- *.png
|           `-- trans/
|-- net/
|-- requirements.txt
`-- README.md
```

The loader pairs each hazy image with its ground-truth image using the numeric
identifier before the first underscore. For example,
`1000_0.74_1.6.png` is paired with `1000.png`. The `trans` directories are not
required by the current training pipeline.

## Environment Setup

The project has been tested on Windows with an NVIDIA GPU and a CUDA-enabled
PyTorch installation.

Create and prepare the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA GPU training, install the CUDA build of PyTorch. The following
command installs the CUDA 13.0 build used during this project:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Verify that PyTorch can access the GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

## Training

Run training from the repository root:

```powershell
.\.venv\Scripts\python.exe net\main.py --data_root F:\FFA-Net\data --trainset haze4k_train --testset haze4k_test --blocks 20 --gps 3 --bs 2 --crop_size 240 --lr 0.0001 --steps 100000 --eval_step 5000
```

CUDA is selected automatically when a compatible GPU is available. Depending
on GPU memory, `--bs` can be increased or reduced. Haze4K images have different
dimensions, so random cropping is enabled by default for batched training.

Checkpoints are saved using the following naming convention:

```text
trained_models/haze4k_train_ffa_3_20.pk
```

The checkpoint stores the model parameters, current training step, loss
history, and the best PSNR and SSIM values.

New checkpoints also include Adam optimizer state. The original best-model
filename and rule (both PSNR and SSIM must improve) are retained. A companion
`*_last.pk` checkpoint records the latest training progress at each evaluation
and at the end of a run. Resume prefers this latest checkpoint and falls back to
the original best-model file. Old checkpoints without optimizer state remain
loadable. Use `--no_resume` or `--resume False` to train from scratch.

`--model_dir` accepts a checkpoint directory or an explicit checkpoint filename.
`--device cpu` and `--device cuda:0` are respected; the default is automatic
selection. Relative data, checkpoint and output paths resolve from the repository
root, regardless of the current working directory.

## Inference

Run inference on the complete Haze4K test split:

```powershell
.\.venv\Scripts\python.exe net\test.py --task haze4k --test_imgs data\Haze4K\test\haze --output_dir samples\haze4k_predictions
```

Run inference on only the first image as a quick check:

```powershell
.\.venv\Scripts\python.exe net\test.py --task haze4k --test_imgs data\Haze4K\test\haze --output_dir samples\haze4k_check --limit 1
```

Use `--model_dir` to load a checkpoint from a custom location:

```powershell
.\.venv\Scripts\python.exe net\test.py --task haze4k --model_dir net\trained_models\haze4k_train_ffa_3_20.pk --test_imgs data\Haze4K\test\haze
```

Predicted images are saved with the `_FFA.png` suffix. Add `--show` to display
each hazy input and prediction during inference.

The original script commands remain available. Module entry points work too:

```powershell
.\.venv\Scripts\python.exe -m net.main --data_root data --bs 2
.\.venv\Scripts\python.exe -m net.test --task haze4k --test_imgs data/Haze4K/test/haze --limit 1
.\.venv\Scripts\python.exe -m net.evaluate_samples --pred_dir samples/haze4k_predictions --gt_dir data/Haze4K/test/gt --csv samples/metrics.csv --plot samples/metrics.png
```

## Results Comparison

These three Haze4K test examples use existing predictions from
`samples/haze4k_test_check/`. The matching hazy inputs and ground-truth images
come from `data/Haze4K/test/`. The images below are unchanged copies stored in
`docs/assets/results/` so the comparison is visible on GitHub.

| Sample | Hazy Input | FFA-Net Output | Ground Truth |
| --- | --- | --- | --- |
| 100 — city | <img src="docs/assets/results/100_hazy.png" alt="Hazy city scene, sample 100" width="220"> | <img src="docs/assets/results/100_ffa.png" alt="FFA-Net city prediction, sample 100" width="220"> | <img src="docs/assets/results/100_gt.png" alt="Ground-truth city scene, sample 100" width="220"> |
| 137 — road | <img src="docs/assets/results/137_hazy.png" alt="Hazy road scene, sample 137" width="220"> | <img src="docs/assets/results/137_ffa.png" alt="FFA-Net road prediction, sample 137" width="220"> | <img src="docs/assets/results/137_gt.png" alt="Ground-truth road scene, sample 137" width="220"> |
| 1000 — indoor | <img src="docs/assets/results/1000_hazy.png" alt="Hazy indoor scene, sample 1000" width="220"> | <img src="docs/assets/results/1000_ffa.png" alt="FFA-Net indoor prediction, sample 1000" width="220"> | <img src="docs/assets/results/1000_gt.png" alt="Ground-truth indoor scene, sample 1000" width="220"> |

### Metrics for the Displayed Examples

Both the hazy input and FFA output are compared with the matching ground truth
using `net/metrics.py`. Metrics are computed on the saved RGB images at their
original resolution, with pixel values in [0, 1]. Higher PSNR and SSIM indicate
closer agreement with the ground truth.

| Sample | Hazy PSNR (dB) | FFA PSNR (dB) | Hazy SSIM | FFA SSIM |
| --- | ---: | ---: | ---: | ---: |
| 100 | 15.61 | 17.48 | 0.7223 | 0.8884 |
| 137 | 8.20 | 16.35 | 0.6581 | 0.8809 |
| 1000 | 21.35 | 20.32 | 0.9400 | 0.9229 |
| Mean of these 3 examples | 15.05 | 18.05 | 0.7735 | 0.8974 |

These are illustrative examples, rather than a full-test-set benchmark.
Samples 100 and 137 improve on both metrics; sample 1000 scores below its
hazy input on both, showing that dehazing does not improve every image.
The source filenames and full-precision scores are available in
[the comparison CSV](docs/assets/results/metrics.csv).

## Project Structure

```text
net/
|-- main.py              Training CLI entry point
|-- test.py              Inference CLI entry point
|-- evaluate_samples.py  Saved-image evaluation, CSV and chart export
|-- option.py            Explicit command-line configuration
|-- data_utils.py        Paired datasets, preprocessing and loader factories
|-- engine.py            Training and validation loop
|-- runtime.py           Project paths, devices and checkpoint compatibility
|-- metrics.py           PSNR and SSIM calculations
`-- models/
    |-- FFA.py           FFA-Net architecture
    `-- PerceptualLoss.py
```

Importing workflow modules does not parse CLI arguments, scan datasets, create
output directories, or start inference. Only the selected datasets are loaded.
Training and inference share the original input normalization constants.
Images smaller than the requested training crop are resized together before
cropping, avoiding the original unbounded image-resampling loop.

See the architecture notes in [English](docs/architecture.md) or
[Chinese](docs/architecture_cn.md) for module responsibilities.

## Regression Checks

Run the small CPU checks without downloading data or weights:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks cover paired augmentation, Haze4K/RESIDE loading, legacy
DataParallel checkpoints, optimizer restoration, batch traversal, import
behavior, and saved-image metrics.

Datasets, weights, virtual environments, predictions, experiment histories,
local backups and archives are excluded by `.gitignore`. Source figures in
`fig/` and checkpoint README files remain versioned.
The selected comparison images and metrics under `docs/assets/results/` are
also versioned.

## Original FFA-Net

This project builds upon:

> Qin, X., Wang, Z., Bai, Y., Xie, X., and Jia, H.
> "FFA-Net: Feature Fusion Attention Network for Single Image Dehazing."
> Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Original paper: <https://arxiv.org/abs/1911.07559>

```bibtex
@inproceedings{qin2020ffa,
  title={FFA-Net: Feature Fusion Attention Network for Single Image Dehazing},
  author={Qin, Xu and Wang, Zhilin and Bai, Yuanchao and Xie, Xiaodong and Jia, Huizhu},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={07},
  pages={11908--11915},
  year={2020}
}
```

## Academic Use

This repository is intended for research and coursework associated with the
COSC428 Research Project. The original FFA-Net implementation, paper, and the
Haze4K dataset should be cited where appropriate.
