import os
from PIL import Image

BASE_DIR = r"D:\ECG_Project\data\ecg_images"

classes = ["Normal", "Abnormal", "MI"]

total_bad = 0

for split in ["train", "val"]:
    print(f"\n===== {split.upper()} =====")

    for cls in classes:
        folder = os.path.join(BASE_DIR, split, cls)

        count = 0
        bad = 0

        for file in os.listdir(folder):
            path = os.path.join(folder, file)

            try:
                img = Image.open(path)
                img.verify()
                count += 1

            except Exception:
                bad += 1
                print("Corrupt:", path)

        total_bad += bad

        print(f"{cls}: {count} images | Corrupt: {bad}")

print("\nTotal corrupt images:", total_bad)