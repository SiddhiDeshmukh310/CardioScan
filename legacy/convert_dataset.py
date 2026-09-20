"""
convert_dataset.py  –  YOLO labels → folder-per-class structure
===============================================================
Reads YOLO .txt label files, takes the MOST COMMON class ID
in each file (not just the first line — more robust), then
copies the image into the matching class folder.

Before:  data/ecg_images/images/train/0_3by4.jpg
         data/ecg_images/labels/train/0_3by4.txt

After:   data/ecg_images/train/class0/0_3by4.jpg
         data/ecg_images/val/class1/51_12by1.jpg
         ...

NOTE: Update CLASS_MAP below once you know your condition names.
      Current mapping is a placeholder — run this script first,
      then rename the folders to meaningful condition names.
"""

import os
import shutil
from collections import Counter

# ── CONFIG — update CLASS_MAP when you know your condition names ──
CLASS_MAP = {
    "0": "class0",   # e.g. replace with "Normal"
    "1": "class1",   # e.g. replace with "Abnormal"
    "2": "class2",   # e.g. replace with "AFib"
    # add more if needed
}

SPLITS = {
    "train": {
        "image_dir":  "data/ecg_images/images/train",
        "label_dir":  "data/ecg_images/labels/train",
        "output_dir": "data/ecg_images/train",
    },
    "val": {
        "image_dir":  "data/ecg_images/images/val",
        "label_dir":  "data/ecg_images/labels/val",
        "output_dir": "data/ecg_images/val",
    },
}

EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


def get_dominant_class(label_path: str) -> str | None:
    """
    Read all lines in a YOLO label file and return the most
    frequent class ID. Falls back to first line if only 1 line.
    Returns None if file is empty or unreadable.
    """
    try:
        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return None
        class_ids = [l.split()[0] for l in lines if l.split()]
        # Return most common class
        return Counter(class_ids).most_common(1)[0][0]
    except Exception as e:
        print(f"  ⚠ Could not read {label_path}: {e}")
        return None


def find_image(image_dir: str, stem: str) -> str | None:
    """Find image file with any supported extension."""
    for ext in EXTENSIONS:
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


def convert_split(split_name: str, cfg: dict):
    image_dir  = cfg["image_dir"]
    label_dir  = cfg["label_dir"]
    output_dir = cfg["output_dir"]

    if not os.path.exists(label_dir):
        print(f"  ⚠ Label dir not found: {label_dir} — skipping {split_name}")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    print(f"\n── {split_name.upper()} ({len(label_files)} label files) ──")

    counts   = Counter()
    skipped  = 0

    for fname in label_files:
        label_path = os.path.join(label_dir, fname)
        stem       = os.path.splitext(fname)[0]

        class_id   = get_dominant_class(label_path)
        if class_id is None:
            skipped += 1
            continue

        if class_id not in CLASS_MAP:
            print(f"  ⚠ Unknown class_id '{class_id}' in {fname} — skipping")
            skipped += 1
            continue

        img_path = find_image(image_dir, stem)
        if img_path is None:
            print(f"  ⚠ No image found for {stem} — skipping")
            skipped += 1
            continue

        dest_folder = os.path.join(output_dir, CLASS_MAP[class_id])
        os.makedirs(dest_folder, exist_ok=True)

        dest_path = os.path.join(dest_folder, os.path.basename(img_path))
        shutil.copy2(img_path, dest_path)
        counts[CLASS_MAP[class_id]] += 1

    print(f"  ✅ Copied: {dict(counts)}")
    if skipped:
        print(f"  ⚠ Skipped: {skipped} files")


def main():
    print("═══ ECG Dataset Conversion ═══")
    print(f"Class map: {CLASS_MAP}\n")

    for split_name, cfg in SPLITS.items():
        convert_split(split_name, cfg)

    print("\n✅ Conversion complete.")
    print("   Next step: run train_model.py")
    print("\n   ⚠ IMPORTANT: Once you identify what class0/class1/class2 mean,")
    print("   update CLASS_MAP in this file and re-run to get meaningful folder names.")


if __name__ == "__main__":
    main()