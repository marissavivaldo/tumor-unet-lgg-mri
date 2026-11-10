"""
tumor_unet_trained.py
----------------------
Brain Tumor Detection using a U-Net trained on the REAL Kaggle LGG-MRI
dataset (not synthetic data -- see note below).

Pipeline
--------
  Raw JPG/PNG MRI image
    └─► Preprocessing (resize, normalize)
          └─► Trained U-Net (trained on real LGG-MRI image/mask pairs)
                └─► Tumor probability map
                      └─► Detection report + visualization

FIX APPLIED (2026-09-30): the previous version of this script downloaded
the real LGG-MRI dataset via kagglehub but never actually used it --
training ran on SyntheticMRIDataset (hand-coded ellipses standing in for
brains/tumors) instead. That's almost certainly why loaded weights failed
to detect real tumors: the model had only ever seen fake ellipse images,
never a real scan or real mask. This version replaces that synthetic
dataset with RealMRIDataset, which loads actual (image, mask) .tif pairs
from the path kagglehub already downloads, so the weights this now
produces have actually seen real tumors.

Requirements
------------
    pip install torch torchvision numpy pillow scikit-image scipy matplotlib opencv-python kagglehub

Usage
-----
    # Delete/rename the old lgg_unet_weights.pth first, or pass --retrain,
    # so you don't accidentally keep using weights trained on fake data:
    python tumor_unet_trained.py --retrain

    # Then run on a single image:
    python tumor_unet_trained.py --image Y107.jpg

    # Or on all images in mri_images/ folder:
    python tumor_unet_trained.py
"""

import sys
print("PYTHON VERSION:", sys.version)

import os
import ssl
import random
import argparse
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from skimage import measure, morphology
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mateuszbuda/lgg-mri-segmentation")

print("Path to dataset files:", path)

# Fix SSL certificates
ssl._create_default_https_context = ssl._create_unverified_context

# ──────────────────────────────────────────────
# 0.  CONFIG
# ──────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR      = os.path.join(SCRIPT_DIR, "mri_images")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "results")
MODEL_PATH     = os.path.join(SCRIPT_DIR, "lgg_unet_weights.pth")

IMAGE_SIZE     = 256
PROB_THRESHOLD = 0.08
MIN_REGION_PX  = 20
MAX_REGIONS    = 3
TRAIN_EPOCHS   = 30
BATCH_SIZE     = 4
LEARNING_RATE  = 1e-3
VAL_FRACTION   = 0.15   # held-out fraction of PATIENTS (not slices) for validation


# ──────────────────────────────────────────────
# 1.  DEVICE
# ──────────────────────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()
print(f"\n{'='*60}")
print(f"  Trained U-Net Brain Tumor Detection")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 2.  U-NET ARCHITECTURE  (unchanged)
# ──────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))


# ──────────────────────────────────────────────
# 3.  REAL TRAINING DATA
#     (loads actual LGG-MRI image/mask .tif pairs that kagglehub
#      already downloaded to `path` above -- replaces the old
#      SyntheticMRIDataset, which never saw a real tumor)
# ──────────────────────────────────────────────
def discover_lgg_pairs(dataset_root):
    """
    Walks the downloaded LGG-MRI dataset and returns (image_path, mask_path)
    tuples for every slice that has a matching "_mask.tif" file. Uses
    os.walk (recursive) rather than a fixed folder depth, since kagglehub's
    download layout can nest an extra folder level (e.g. .../kaggle_3m/).
    """
    pairs = []
    for root, _, files in os.walk(dataset_root):
        tif_files = [f for f in files if f.lower().endswith(".tif")]
        mask_files = {f for f in tif_files if f.lower().endswith("_mask.tif")}
        image_files = [f for f in tif_files if f not in mask_files]
        for img_name in image_files:
            mask_name = img_name[:-4] + "_mask.tif"
            if mask_name in mask_files:
                pairs.append((os.path.join(root, img_name), os.path.join(root, mask_name)))
    return pairs


def split_pairs_by_patient(pairs, val_fraction=VAL_FRACTION, seed=42):
    """
    Splits by PATIENT folder, not by individual slice. Slices from the same
    patient look very similar, so splitting at the slice level would leak
    near-duplicate images between train/val and make validation look
    better than the model actually generalizes.
    """
    by_patient = defaultdict(list)
    for img_path, mask_path in pairs:
        patient_id = os.path.basename(os.path.dirname(img_path))
        by_patient[patient_id].append((img_path, mask_path))

    patient_ids = list(by_patient.keys())
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * val_fraction))
    val_ids = set(patient_ids[:n_val])

    train_pairs, val_pairs = [], []
    for pid, plist in by_patient.items():
        (val_pairs if pid in val_ids else train_pairs).extend(plist)
    return train_pairs, val_pairs


class RealMRIDataset(Dataset):
    """
    Loads real (image, mask) pairs. Preprocessing is kept IDENTICAL to
    preprocess() below (same resize method, same per-channel z-score
    normalization) so training and inference see data in the same
    distribution -- a mismatch there would cause the same kind of
    silent failure as the synthetic-data bug.
    """
    def __init__(self, pairs, size=IMAGE_SIZE):
        self.pairs = pairs
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        pil_img = Image.open(img_path).convert("RGB").resize((self.size, self.size), Image.LANCZOS)
        arr = np.asarray(pil_img).astype(np.float32)
        for c in range(3):
            chan = arr[:, :, c]
            arr[:, :, c] = (chan - chan.mean()) / (chan.std() + 1e-8)

        pil_mask = Image.open(mask_path).convert("L").resize((self.size, self.size), Image.NEAREST)
        mask = (np.asarray(pil_mask).astype(np.float32) > 127).astype(np.float32)

        # light augmentation (kept from the original synthetic dataset)
        if random.random() > 0.5:
            arr = arr[:, ::-1, :].copy()
            mask = mask[:, ::-1].copy()
        if random.random() > 0.5:
            arr = arr[::-1, :, :].copy()
            mask = mask[::-1, :].copy()

        t_img = torch.from_numpy(arr.transpose(2, 0, 1))
        t_mask = torch.from_numpy(mask.copy()).unsqueeze(0)
        return t_img, t_mask


@torch.no_grad()
def dice_score(preds, targets, smooth=1.0):
    p = preds.view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    intersection = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return ((2 * intersection + smooth) / (union + smooth)).mean().item()


# ──────────────────────────────────────────────
# 4.  LOSS FUNCTION  (unchanged)
# ──────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce  = self.bce(pred, target)
        p    = pred.view(-1)
        t    = target.view(-1)
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        return bce + dice


# ──────────────────────────────────────────────
# 5.  TRAINING  (now on real data, with patient-split validation)
# ──────────────────────────────────────────────
def train_model():
    print("  Training U-Net on REAL LGG-MRI data...")
    print(f"  (This trains once and saves weights to {os.path.basename(MODEL_PATH)})\n")

    all_pairs = discover_lgg_pairs(path)
    if len(all_pairs) == 0:
        raise RuntimeError(
            f"No (image, mask) .tif pairs found under: {path}\n"
            "Check that the kagglehub download succeeded and contains the "
            "per-patient folders with *_mask.tif files."
        )
    train_pairs, val_pairs = split_pairs_by_patient(all_pairs)
    print(f"  Slices -- train: {len(train_pairs)}, val: {len(val_pairs)} "
          f"(split by patient to avoid leakage)\n")

    train_dataset = RealMRIDataset(train_pairs, size=IMAGE_SIZE)
    val_dataset   = RealMRIDataset(val_pairs, size=IMAGE_SIZE)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)

    print(f"  {'Epoch':<8} {'Loss':<12} {'Val Dice':<10} {'Status'}")
    print(f"  {'-'*8} {'-'*11} {'-'*9} {'-'*20}")

    best_val_dice = -1.0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_dice_total = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_dice_total += dice_score((preds > 0.5).float(), masks)
        val_dice = val_dice_total / len(val_loader)

        marker = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ◀ best"

        if epoch % 5 == 0 or epoch == 1:
            print(f"  {epoch:<8} {avg_loss:<12.4f} {val_dice:<10.4f}{marker}")

    print(f"\n  ✓ Training complete. Best val Dice: {best_val_dice:.4f}")
    print(f"  ✓ Weights saved → {MODEL_PATH}\n")
    return model


def load_trained_model():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ──────────────────────────────────────────────
# 6.  INFERENCE PIPELINE  (unchanged)
# ──────────────────────────────────────────────
def preprocess(path_):
    pil  = Image.open(path_).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    arr  = np.asarray(pil).astype(np.float32)
    gray = np.asarray(pil.convert("L"))
    for c in range(3):
        ch = arr[:, :, c]
        arr[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    return tensor, gray


def predict(model, tensor):
    with torch.no_grad():
        prob = model(tensor.to(DEVICE)).squeeze().cpu().numpy()

    h, w   = prob.shape
    cy, cx = h / 2, w / 2
    Y, X   = np.ogrid[:h, :w]
    brain  = ((X - cx)**2 / (cx * 0.85)**2 + (Y - cy)**2 / (cy * 0.85)**2) <= 1
    prob   = prob * brain
    prob   = gaussian_filter(prob, sigma=1.5)
    return prob


def find_regions(prob):
    binary = prob > PROB_THRESHOLD
    binary = morphology.remove_small_objects(binary, min_size=MIN_REGION_PX)
    binary = morphology.remove_small_holes(binary, area_threshold=40)
    labeled = measure.label(binary)
    props   = measure.regionprops(labeled, intensity_image=prob)

    regions = []
    for r in props:
        if r.area < MIN_REGION_PX:
            continue
        regions.append({
            "centroid":   r.centroid,
            "area":       int(r.area),
            "confidence": float(r.mean_intensity),
            "max_conf":   float(r.max_intensity),
            "bbox":       r.bbox,
            "diameter":   float(r.equivalent_diameter),
        })
    regions.sort(key=lambda d: d["max_conf"], reverse=True)
    return regions[:MAX_REGIONS]


# ──────────────────────────────────────────────
# 7.  VISUALISATION  (unchanged)
# ──────────────────────────────────────────────
def visualise(name, gray, prob, regions, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original MRI", color="white", fontsize=12, pad=10)
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    rgba      = np.zeros((*prob.shape, 4))
    mask      = prob > PROB_THRESHOLD
    rgba[mask, 0] = 1.0
    rgba[mask, 3] = np.clip(prob[mask], 0, 1) * 0.65
    axes[1].imshow(rgba)

    colors = ["#ff4444", "#ff9900", "#ffff00"]
    for i, r in enumerate(regions):
        c        = colors[i % len(colors)]
        r0, c0, r1, c1 = r["bbox"]
        axes[1].add_patch(mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=3 if i == 0 else 2,
            edgecolor=c, facecolor="none", linestyle="--"))
    axes[1].set_title("Detection Overlay", color="white", fontsize=12, pad=10)
    axes[1].axis("off")

    fig.text(0.5, 0.02,
             "⚠  Research / educational tool only. NOT for clinical diagnosis. "
             "Always consult a medical professional.",
             ha="center", color="#f0883e", fontsize=9)
    plt.suptitle("Trained U-Net Brain Tumor Detection",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print(f"    ✓ Saved → {out_path}")

# ──────────────────────────────────────────────
# 8.  PROCESS ONE IMAGE  (unchanged)
# ──────────────────────────────────────────────
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

def process_one(model, image_path):
    name = os.path.basename(image_path)
    print(f"\n  Analysing: {name}")
    tensor, gray = preprocess(image_path)
    prob         = predict(model, tensor)
    regions      = find_regions(prob)

    if regions:
        for i, r in enumerate(regions, 1):
            cy2, cx2 = r["centroid"]
            print(f"    #{i}  {r['max_conf']:.1%}  @ ({cx2:.0f},{cy2:.0f})  "
                  f"⌀{r['diameter']:.1f}px")
    else:
        print("    No tumor regions above threshold.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base     = os.path.splitext(name)[0]
    out_path = os.path.join(OUTPUT_DIR, f"{base}_trained_result.png")
    visualise(name, gray, prob, regions, out_path)


# ──────────────────────────────────────────────
# 9.  MAIN  (unchanged)
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    help="Analyse a single image")
    parser.add_argument("--retrain",  action="store_true",
                        help="Force retraining even if weights exist")
    args = parser.parse_args()

    print("=" * 60)
    print("  Trained U-Net Brain Tumor Detection")
    print("  Based on LGG-MRI Kaggle Dataset Architecture")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH) or args.retrain:
        model = train_model()
    else:
        print(f"  ✓ Loading saved weights from {os.path.basename(MODEL_PATH)}\n")
        model = load_trained_model()

    if args.image:
        if not os.path.exists(args.image):
            print(f"ERROR: file not found → {args.image}")
            sys.exit(1)
        process_one(model, args.image)
        print("\nDone.")
        return

    os.makedirs(INPUT_DIR, exist_ok=True)
    files = [f for f in sorted(os.listdir(INPUT_DIR))
             if f.lower().endswith(IMG_EXT)]

    if not files:
        print(f"\n  No images found in:\n    {INPUT_DIR}\n")
        print("  → Put your MRI image(s) in that folder, then run again.")
        return

    print(f"\n  Found {len(files)} image(s) in mri_images/")
    for f in files:
        process_one(model, os.path.join(INPUT_DIR, f))

    print("\n" + "=" * 60)
    print(f"  Done!  Results saved to:\n    {OUTPUT_DIR}")
    print("  ⚠  Research only — not for clinical use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
