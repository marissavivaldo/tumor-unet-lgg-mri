# Brain Tumor Segmentation with a U-Net (LGG-MRI)

A standard U-Net implemented in PyTorch for brain MRI tumor segmentation,
built as a learning project to understand the encoder–decoder architecture
in depth.

## Results (held-out validation, patient-level split)

- Train / validation: 6,924 / 934 slices (15% of *patients* held out, so no
  near-duplicate slices from the same patient leak into validation)
- **Pooled Dice: 0.77** (raw model, threshold 0.5)
- Mean per-slice Dice on tumor-positive slices only: 0.67 (n = 322)
- Best checkpoint: epoch 27 of 30

Note: the training script's built-in per-slice Dice (0.88) averages in the
many tumor-free slices and is inflated; the pooled and tumor-only figures
above are the defensible ones.

## What this project taught / demonstrates

- Encoder–decoder U-Net with skip connections, 256×256 inputs
- Dice + BCE combined loss, cosine-annealing learning-rate schedule
- Patient-level train/validation splitting to avoid data leakage
- Debugging a silent data bug: an earlier version downloaded the real
  dataset but trained on synthetic images. Fixed 2026-09-30 by replacing
  the synthetic dataset with a loader for the real image/mask pairs.

## Dataset

Kaggle LGG-MRI Segmentation dataset (Mateusz Buda) — downloaded
automatically via `kagglehub`. The dataset itself is **not** included in
this repository. You will need Kaggle credentials configured for
`kagglehub` to download it.

## Usage

```
pip install -r requirements.txt
python tumor_unet_real_data.py --retrain     # train on real data (30 epochs)
python tumor_unet_real_data.py --image Y107.jpg
python tumor_unet_real_data.py               # all images in mri_images/
```

## ⚠ Research / educational use only — not for clinical diagnosis.
