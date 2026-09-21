import argparse
import os
import re
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

def load_split_dataframe(data_dir: str, split_file: str):
    if not os.path.exists(split_file):
        raise FileNotFoundError(f'Split file not found: {split_file}')

    split_df = pd.read_csv(split_file)
    split_map = dict(zip(split_df['ecg_id'], split_df['split']))

    records = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                cls_name = os.path.basename(root)
                if cls_name not in ['Normal', 'Abnormal', 'MI']:
                    continue
                m = re.search(r'(\d+)', f)
                if m:
                    ecg_id = int(m.group(1))
                    split = split_map.get(ecg_id, 'unmapped')
                    full_path = os.path.join(root, f)
                    records.append({'filepath': full_path, 'ecg_id': ecg_id, 'class': cls_name, 'split': split})

    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet ECG Classifier with PTB-XL strat_fold split')
    parser.add_argument('--data_dir', type=str, default='data/ecg_images', help='Base directory for ECG images')
    parser.add_argument('--model_path', type=str, default='model/best_efficientnet.h5', help='Output model path')
    parser.add_argument('--split_file', type=str, default='splits/split.csv', help='CSV containing leak-free strat_fold split')
    args = parser.parse_args()

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16

    if not os.path.exists(args.split_file) or not os.path.exists(args.data_dir):
        print(f'Data or split file missing ({args.data_dir}, {args.split_file}). Ready for Kaggle/Colab execution.')
        return

    df = load_split_dataframe(args.data_dir, args.split_file)
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']

    print(f'Leak-free Train samples: {len(df_train)} | Val samples: {len(df_val)}')

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_dataframe(
        df_train,
        x_col='filepath',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_data = val_gen.flow_from_dataframe(
        df_val,
        x_col='filepath',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(len(train_data.class_indices), activation='softmax')(x)

    model = Model(base.input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        ModelCheckpoint(args.model_path, save_best_only=True, monitor='val_accuracy')
    ]

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_data.classes),
        y=train_data.classes
    )
    class_weights = dict(enumerate(class_weights))

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=15,
        callbacks=callbacks,
        class_weight=class_weights
    )
    print('\nTraining Finished')

if __name__ == '__main__':
    main()
