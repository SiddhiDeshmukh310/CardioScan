import argparse
import os
import random
import cv2
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Quick Inference Test for ECG Model')
    parser.add_argument('--model_path', type=str, default='model/best_efficientnet.h5', help='Path to model file')
    parser.add_argument('--data_dir', type=str, default='data/ecg_images/train', help='Path to image folder')
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f'Model file not found at {args.model_path}. Skipping quick test.')
        return

    model = tf.keras.models.load_model(args.model_path)
    CLASS_NAMES = {0: 'Abnormal', 1: 'MI', 2: 'Normal'}

    for cls in ['Normal', 'Abnormal', 'MI']:
        folder = os.path.join(args.data_dir, cls)
        if not os.path.exists(folder):
            continue
        print(f'\n===== {cls} =====')
        files = os.listdir(folder)
        samples = random.sample(files, min(3, len(files)))

        for file in samples:
            path = os.path.join(folder, file)
            img = cv2.imread(path)
            img = cv2.resize(img, (224, 224))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, 0)
            pred = model.predict(img, verbose=0)[0]
            idx = np.argmax(pred)
            print(file, '->', CLASS_NAMES.get(idx, str(idx)), pred)

if __name__ == '__main__':
    main()
