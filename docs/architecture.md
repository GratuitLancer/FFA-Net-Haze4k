# Architecture Notes

[Chinese original](architecture_cn.md)

This refactor preserves the PyTorch framework, the original `net/` directory,
and the FFA model. The three residual groups, channel attention, pixel attention,
feature fusion, and parameter names remain unchanged. The implementations of
`FFA.py`, `PerceptualLoss.py`, and `metrics.py` have not been modified.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `net/main.py` | Training entry point; assembles datasets, model, losses, and optimizer |
| `net/option.py` | Explicitly parses and validates training arguments |
| `net/data_utils.py` | Haze4K/RESIDE pairing, cropping, augmentation, normalization, and loader factories |
| `net/engine.py` | Training, validation, learning-rate scheduling, and training resume |
| `net/runtime.py` | Shared project paths, device selection, and checkpoint I/O |
| `net/test.py` | Image inference entry point |
| `net/evaluate_samples.py` | PSNR/SSIM evaluation of saved predictions, plus CSV and chart export |
| `net/models/` | Original FFA network and perceptual loss |

Configuration parsing, dataset scanning, and directory creation occur only
after explicit calls; importing modules does not start a task. Original commands
such as `python net/main.py` remain available, alongside module entry points such
as `python -m net.main`. Relative paths resolve from the project root. Inference
also supports legacy input and checkpoint paths under `net/`.

## Training Behavior and Compatibility

- Only the selected training and validation datasets are constructed. Haze4K
  files are still paired by the identifier before the first underscore.
- The training iterator is recreated after an epoch ends, avoiding the previous
  behavior of obtaining the first batch from a new iterator at every step.
- Input normalization, L1 loss, the optional perceptual-loss weight of 0.04, and
  the cosine learning-rate schedule retain their original settings.
- Validation uses the original PSNR/SSIM implementations. Both metrics must
  improve before a new best-model checkpoint is saved.
- Best-model checkpoints retain their original `*.pk` filenames. A new
  `*_last.pk` checkpoint stores the latest progress and optimizer state.
- Resume prefers the latest checkpoint. Legacy checkpoints remain loadable even
  when they do not contain optimizer state.
- Both plain model weights and DataParallel weights with the `module.` prefix
  are supported.
- Image pairs smaller than the requested crop size are resized together,
  preventing the original image-resampling loop from running indefinitely.
- With cropping disabled and a batch size greater than 1, input images must
  still have matching dimensions.

## Local Validation

`python -m unittest discover -s tests -v` uses small temporary images and the
CPU to check import behavior, image pairing, synchronized augmentation, argument
parsing, checkpoint compatibility, consecutive batches, optimizer restoration,
and saved-image evaluation. It does not download datasets or pretrained networks.
