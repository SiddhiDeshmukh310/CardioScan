"""
fix_dataset.py  —  Audit and fix the ECG dataset structure
===========================================================
Run this BEFORE train_model.py.

Problems it fixes:
  1. Merges duplicate folders (class0 + normal → Normal)
  2. Reports class distribution
  3. Warns about classes with too few images
  4. Shows you what class IDs actually exist in your labels
     so you can update CLASS_MAP correctly
"""

import os
import shutil
from collections import Counter

# ── STEP 1: Show what's currently in your dataset ─────────────────
print("═══ CURRENT DATASET STATE ═══\n")
for split in ["train", "val"]:
    base = f"data/ecg_images/{split}"
    if not os.path.exists(base):
        print(f"  ⚠ {base} not found")
        continue
    print(f"{split}/")
    total = 0
    for cls in sorted(os.listdir(base)):
        path = os.path.join(base, cls)
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
            print(f"  {cls}/  →  {n} images")
            total += n
    print(f"  TOTAL: {total}\n")

# ── STEP 2: Show what class IDs actually exist in your labels ──────
print("═══ CLASS IDs IN LABEL FILES ═══\n")
for split in ["train", "val"]:
    label_dir = f"data/ecg_images/labels/{split}"
    if not os.path.exists(label_dir):
        continue
    counter = Counter()
    files   = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    for fname in files:
        path = os.path.join(label_dir, fname)
        try:
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            if lines:
                # first class id in file = image-level label
                counter[lines[0].split()[0]] += 1
        except Exception:
            pass
    print(f"{split}/labels class distribution:")
    for cls_id, count in sorted(counter.items()):
        print(f"  class_id '{cls_id}': {count} images")
    print()

# ── STEP 3: Merge duplicate folders ───────────────────────────────
# class0 and normal are the same — merge into one clean folder
print("═══ MERGING DUPLICATE FOLDERS ═══\n")

MERGE_MAP = {
    "train": [
        # (source_folders_to_merge, destination_folder_name)
        (["class0", "normal"], "Normal"),
        (["class1"], "Abnormal"),   # only if class1 has enough images
        (["class2"], "AFib"),
    ],
    "val": [
        (["class0", "normal"], "Normal"),
        (["class1"], "Abnormal"),
        (["class2"], "AFib"),
    ],
}

MIN_IMAGES = 10   # skip destination if source has fewer than this

for split, merges in MERGE_MAP.items():
    base = f"data/ecg_images/{split}"
    if not os.path.exists(base):
        continue

    for sources, dest_name in merges:
        dest_path = os.path.join(base, dest_name)
        moved = 0

        for src_name in sources:
            src_path = os.path.join(base, src_name)
            if not os.path.exists(src_path):
                continue

            imgs = [f for f in os.listdir(src_path)
                    if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))]

            if len(imgs) < MIN_IMAGES:
                print(f"  ⚠ Skipping {split}/{src_name} → only {len(imgs)} images (< {MIN_IMAGES})")
                continue

            os.makedirs(dest_path, exist_ok=True)
            for img in imgs:
                src_file  = os.path.join(src_path, img)
                # avoid name collisions
                dest_file = os.path.join(dest_path, f"{src_name}_{img}")
                shutil.copy2(src_file, dest_file)
                moved += 1

            print(f"  ✅ Merged {split}/{src_name} → {split}/{dest_name}  ({len(imgs)} images)")

        if moved > 0:
            # remove old source folders after successful merge
            for src_name in sources:
                src_path = os.path.join(base, src_name)
                if os.path.exists(src_path) and src_name != dest_name:
                    shutil.rmtree(src_path)
                    print(f"  🗑  Removed old folder: {split}/{src_name}")

# ── STEP 4: Final state ────────────────────────────────────────────
print("\n═══ DATASET AFTER CLEANUP ═══\n")
valid_classes = []
for split in ["train", "val"]:
    base = f"data/ecg_images/{split}"
    if not os.path.exists(base):
        continue
    print(f"{split}/")
    for cls in sorted(os.listdir(base)):
        path = os.path.join(base, cls)
        if os.path.isdir(path):
            n = len([f for f in os.listdir(path)
                     if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
            status = "✅" if n >= MIN_IMAGES else "⚠ TOO FEW"
            print(f"  {cls}/  →  {n} images  {status}")
            if split == "train" and n >= MIN_IMAGES:
                valid_classes.append(cls)
    print()

print("═══ NEXT STEPS ═══\n")
if len(valid_classes) < 2:
    print("⚠ WARNING: You need at least 2 classes with enough images to train.")
    print("  Your label files show only 1 usable class right now.")
    print()
    print("  ACTION REQUIRED:")
    print("  1. Open a few label .txt files from data/ecg_images/labels/train/")
    print("  2. Check what class IDs are in them (0, 1, 2 ...)")
    print("  3. Open convert_dataset.py and update CLASS_MAP to match")
    print("     e.g. {'0': 'Normal', '1': 'AFib', '2': 'MI'}")
    print("  4. Re-run convert_dataset.py")
    print("  5. Re-run this script")
else:
    print(f"✅ {len(set(valid_classes))} valid classes found: {sorted(set(valid_classes))}")
    print("   You can now run:  python train_model.py")