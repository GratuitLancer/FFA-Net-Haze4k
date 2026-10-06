# Results

This page documents the qualitative examples currently committed to the repository.

## Important scope note

The following three samples are **illustrative examples**, not a full Haze4K test-set benchmark.

They are useful for showing:

- strong improvement on difficult haze,
- the visual effect of FFA-Net,
- and one failure case where enhancement does not improve the reference metrics.

That last case is intentionally retained instead of cherry-picking only successful outputs.

## Side-by-side comparisons

### Sample 100

| Hazy input | FFA-Net output | Ground truth |
|---|---|---|
| <img src="assets/results/100_hazy.png" alt="Hazy input for sample 100" width="300"> | <img src="assets/results/100_ffa.png" alt="FFA-Net output for sample 100" width="300"> | <img src="assets/results/100_gt.png" alt="Ground truth for sample 100" width="300"> |

- Hazy PSNR: **15.61 dB**
- FFA PSNR: **17.48 dB**
- Hazy SSIM: **0.7223**
- FFA SSIM: **0.8884**

### Sample 137

| Hazy input | FFA-Net output | Ground truth |
|---|---|---|
| <img src="assets/results/137_hazy.png" alt="Hazy input for sample 137" width="300"> | <img src="assets/results/137_ffa.png" alt="FFA-Net output for sample 137" width="300"> | <img src="assets/results/137_gt.png" alt="Ground truth for sample 137" width="300"> |

- Hazy PSNR: **8.20 dB**
- FFA PSNR: **16.35 dB**
- Hazy SSIM: **0.6581**
- FFA SSIM: **0.8809**

This is the strongest example among the three displayed samples.

### Sample 1000

| Hazy input | FFA-Net output | Ground truth |
|---|---|---|
| <img src="assets/results/1000_hazy.png" alt="Hazy input for sample 1000" width="300"> | <img src="assets/results/1000_ffa.png" alt="FFA-Net output for sample 1000" width="300"> | <img src="assets/results/1000_gt.png" alt="Ground truth for sample 1000" width="300"> |

- Hazy PSNR: **21.35 dB**
- FFA PSNR: **20.32 dB**
- Hazy SSIM: **0.9400**
- FFA SSIM: **0.9229**

This sample gets slightly worse on both metrics.

That does not invalidate the method. It illustrates an important limitation of dehazing systems: a network may alter contrast, colour, or local detail even when the original hazy image is already numerically close to the reference.

## Three-example summary

| Sample | Hazy PSNR | FFA PSNR | Δ PSNR | Hazy SSIM | FFA SSIM | Δ SSIM |
|---|---:|---:|---:|---:|---:|---:|
| 100 | 15.61 | 17.48 | +1.87 | 0.7223 | 0.8884 | +0.1661 |
| 137 | 8.20 | 16.35 | +8.15 | 0.6581 | 0.8809 | +0.2228 |
| 1000 | 21.35 | 20.32 | -1.03 | 0.9400 | 0.9229 | -0.0171 |
| Mean | 15.05 | 18.05 | +3.00 | 0.7735 | 0.8974 | +0.1239 |

## Interpretation

The three examples show two different behaviors.

### Severe haze

For samples 100 and 137, FFA-Net substantially improves reconstruction quality.

The gain is especially large for sample 137, where PSNR rises by more than 8 dB and SSIM rises by more than 0.22.

### Already-close input

Sample 1000 begins with relatively high similarity to the ground truth.

The network still transforms the image, but the transformation does not move the result closer to the reference according to PSNR or SSIM.

This is a useful reminder that an enhancement network is not automatically beneficial for every input.

## Source data

The full-precision values are stored in:

```text
docs/assets/results/metrics.csv
```

The original prediction, hazy input, and ground-truth images are stored in:

```text
docs/assets/results/
```

## Reproduce evaluation

For a larger set of generated predictions:

```bash
python -m net.evaluate_samples \
  --pred_dir samples/haze4k_predictions \
  --gt_dir data/Haze4K/test/gt \
  --csv samples/metrics.csv \
  --plot samples/metrics.png
```

A full-test-set evaluation should be reported separately from the three example results above.
