import argparse
import os
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Verify Image Integrity')
    parser.add_argument('--data_dir', type=str, default='data/ecg_images', help='Base directory for ECG images')
    args = parser.parse_args()

    classes = ['Normal', 'Abnormal', 'MI']
    total_bad = 0

    for split in ['train', 'val']:
        print(f'\n===== {split.upper()} =====')
        for cls in classes:
            folder = os.path.join(args.data_dir, split, cls)
            if not os.path.exists(folder):
                print(f'Folder missing: {folder}')
                continue
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
                    print('Corrupt:', path)
            total_bad += bad
            print(f'{cls}: {count} images | Corrupt: {bad}')

    print('\nTotal corrupt images:', total_bad)

if __name__ == '__main__':
    main()
