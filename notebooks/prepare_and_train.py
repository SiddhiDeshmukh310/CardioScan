"""
prepare_and_train.py  —  PTB-XL ECG: Label → Sort → Train
===========================================================
Run from D:\\ECG_Project:
    python prepare_and_train.py

What it does:
  1. Reads ptbxl_database.csv  → maps record_id → condition
  2. Reads scp_statements.csv  → knows which codes are Normal/AFib/MI etc.
  3. Scans your image filenames → extracts record_id
  4. Copies images into class folders (Normal / AFib / Other)
  5. Trains EfficientNetB0 classifier
  6. Saves model/ecg_model.h5 + model/class_names.json
"""

import os, json, shutil
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────
CSV_PATH   = "ptbxl_database.csv"
SCP_PATH   = "scp_statements.csv"      # may not exist — handled below
IMAGE_DIR  = "data/ecg_images/images"  # contains train/ and val/
OUT_DIR    = "data/ecg_images"         # train/ val/ class folders go here
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ecg_model.h5")
NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 16
EPOCHS_FROZEN   = 10
EPOCHS_FINETUNE = 15
LR_FROZEN   = 1e-3
LR_FINETUNE = 1e-4

os.makedirs(MODEL_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
#  STEP 1 — Load PTB-XL CSV and build record_id → condition map
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 1: Reading PTB-XL labels ═══")

df = pd.read_csv(CSV_PATH)
print(f"  Loaded {len(df)} records from {CSV_PATH}")
print(f"  Columns: {list(df.columns[:8])} ...")

# scp_codes column looks like: "{'NORM': 100.0}" or "{'AFIB': 100.0, 'SR': 0.0}"
import ast

def parse_scp(scp_str):
    """Parse scp_codes string → dict of {code: likelihood}"""
    try:
        return ast.literal_eval(scp_str)
    except Exception:
        return {}

df["scp_dict"] = df["scp_codes"].apply(parse_scp)

# ── Load scp_statements to know which codes = rhythm vs diagnostic ──
# We'll classify based on the highest-likelihood code
RHYTHM_MAP = {
    # Normal
    "NORM":  "Normal",
    "SR":    "Normal",       # Sinus Rhythm
    # AFib / Flutter
    "AFIB":  "AFib",
    "AFLT":  "AFib",         # Atrial Flutter
    # MI
    "IMI":   "MI",           # Inferior MI
    "AMI":   "MI",           # Anterior MI
    "LMI":   "MI",           # Lateral MI
    "ILMI":  "MI",
    "ALMI":  "MI",
    "INJAS": "MI",
    "INJAL": "MI",
    "INJIN": "MI",
    "INJLA": "MI",
    # Conduction
    "LBBB":  "Other",
    "RBBB":  "Other",
    "IRBBB": "Other",
    "CLBBB": "Other",
    # ST changes
    "STD_":  "Other",
    "STE_":  "Other",
    # Everything else → Other
}

def get_condition(scp_dict):
    """Return the dominant condition for a record."""
    if not scp_dict:
        return None
    # Sort by likelihood descending
    sorted_codes = sorted(scp_dict.items(), key=lambda x: x[1], reverse=True)
    for code, likelihood in sorted_codes:
        if likelihood >= 50:        # only use high-confidence codes
            cond = RHYTHM_MAP.get(code)
            if cond:
                return cond
    # Fallback: any mapped code
    for code, _ in sorted_codes:
        cond = RHYTHM_MAP.get(code)
        if cond:
            return cond
    return "Other"

df["condition"] = df["scp_dict"].apply(get_condition)

# Build lookup: record_id (int) → condition
id_col = "ecg_id" if "ecg_id" in df.columns else df.columns[0]
record_map = {}
for _, row in df.iterrows():
    rid  = int(row[id_col])
    cond = row["condition"]
    if cond:
        record_map[rid] = cond

print(f"\n  Condition distribution in CSV:")
cond_counts = Counter(record_map.values())
for cond, count in sorted(cond_counts.items()):
    print(f"    {cond}: {count} records")

# ══════════════════════════════════════════════════════════════════
#  STEP 2 — Sort images into class folders
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 2: Sorting images into class folders ═══")

stats = {"train": Counter(), "val": Counter()}
skipped = 0

for split in ["train", "val"]:
    src_dir = os.path.join(IMAGE_DIR, split)
    if not os.path.exists(src_dir):
        print(f"  ⚠ {src_dir} not found — skipping")
        continue

    files = [f for f in os.listdir(src_dir)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"\n  Processing {split}/ ({len(files)} images)...")

    for fname in files:
        # Extract record_id: "583_12by1.jpg" → 583
        stem = Path(fname).stem           # "583_12by1"
        parts = stem.split("_")
        try:
            record_id = int(parts[0])
        except ValueError:
            skipped += 1
            continue

        condition = record_map.get(record_id)
        if condition is None:
            skipped += 1
            continue

        # Copy to destination
        dest_folder = os.path.join(OUT_DIR, split, condition)
        os.makedirs(dest_folder, exist_ok=True)
        src_file  = os.path.join(src_dir, fname)
        dest_file = os.path.join(dest_folder, fname)
        if not os.path.exists(dest_file):      # skip if already copied
            shutil.copy2(src_file, dest_file)
        stats[split][condition] += 1

print(f"\n  Results:")
for split in ["train", "val"]:
    print(f"  {split}/")
    for cond, n in sorted(stats[split].items()):
        print(f"    {cond}: {n} images")
if skipped:
    print(f"  ⚠ Skipped {skipped} images (no matching record in CSV)")

# ══════════════════════════════════════════════════════════════════
#  STEP 3 — Validate we have enough data
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 3: Validating dataset ═══")

train_base = os.path.join(OUT_DIR, "train")
valid_classes = []
for cls in sorted(os.listdir(train_base)):
    path = os.path.join(train_base, cls)
    if not os.path.isdir(path):
        continue
    n = len([f for f in os.listdir(path)
             if f.lower().endswith(('.jpg','.jpeg','.png','.bmp'))])
    status = "✅" if n >= 20 else "⚠ TOO FEW"
    print(f"  train/{cls}: {n} images  {status}")
    if n >= 20:
        valid_classes.append(cls)

if len(valid_classes) < 2:
    print("\n❌ Need at least 2 classes with 20+ images to train.")
    print("   Check that ptbxl_database.csv record IDs match your filenames.")
    exit(1)

print(f"\n✅ Training with {len(valid_classes)} classes: {valid_classes}")

# ══════════════════════════════════════════════════════════════════
#  STEP 4 — Train EfficientNetB0
# ══════════════════════════════════════════════════════════════════
print("\n═══ STEP 4: Training CNN ═══")

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

train_gen = ImageDataGenerator(
    rescale            = 1.0/255,
    rotation_range     = 3,
    width_shift_range  = 0.05,
    height_shift_range = 0.05,
    zoom_range         = 0.05,
    horizontal_flip    = False,
    fill_mode          = 'nearest',
)
val_gen = ImageDataGenerator(rescale=1.0/255)

train_data = train_gen.flow_from_directory(
    train_base,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    shuffle     = True,
    classes     = sorted(valid_classes),
)

val_base  = os.path.join(OUT_DIR, "val")
val_data  = val_gen.flow_from_directory(
    val_base,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    shuffle     = False,
    classes     = sorted(valid_classes),
)

NUM_CLASSES = len(train_data.class_indices)
print(f"\n  Classes : {train_data.class_indices}")
print(f"  Train   : {train_data.samples} images")
print(f"  Val     : {val_data.samples} images")

# Save class names
class_names = {str(v): k for k, v in train_data.class_indices.items()}
with open(NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"  Saved class names → {NAMES_PATH}")

# Class weights
cw = compute_class_weight("balanced",
                          classes=np.unique(train_data.classes),
                          y=train_data.classes)
class_weights = dict(enumerate(cw))
print(f"  Class weights: { {class_names[str(k)]: round(v,2) for k,v in class_weights.items()} }")

# Model
backbone = EfficientNetB0(weights="imagenet", include_top=False,
                          input_shape=(*IMG_SIZE, 3))
backbone.trainable = False

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = backbone(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model   = Model(inputs, outputs)

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=6,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
]

# Phase 1 — frozen backbone
print("\n── Phase 1: Frozen backbone ──")
model.compile(optimizer=tf.keras.optimizers.Adam(LR_FROZEN),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FROZEN,
          class_weight=class_weights, callbacks=callbacks, verbose=1)

# Phase 2 — fine-tune top 30 layers
print("\n── Phase 2: Fine-tuning top layers ──")
backbone.trainable = True
for layer in backbone.layers[:-30]:
    layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(LR_FINETUNE),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINETUNE,
          class_weight=class_weights, callbacks=callbacks, verbose=1)

# Final eval
print("\n═══ Final Evaluation ═══")
model.load_weights(MODEL_PATH)
loss, acc = model.evaluate(val_data, verbose=1)
print(f"\n✅ Best val accuracy : {acc*100:.2f}%")
print(f"✅ Model saved       → {MODEL_PATH}")
print(f"✅ Class names       → {NAMES_PATH}")
print("\n   Now restart your Flask app — it will auto-load the model.")