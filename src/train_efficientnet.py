import argparse
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

def main():
    parser = argparse.ArgumentParser(description='Train EfficientNet ECG Classifier with PTB-XL strat_fold split')
    parser.add_argument('--data_dir', type=str, default='data/ecg_images', help='Base directory for ECG images')
    parser.add_argument('--model_path', type=str, default='model/best_efficientnet.h5', help='Output model path')
    parser.add_argument('--split_file', type=str, default='splits/split.csv', help='CSV containing leak-free strat_fold split')
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=3,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.05
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    if not (os.path.exists(train_dir) and os.path.exists(val_dir)):
        print(f'Data directories missing ({train_dir}, {val_dir}). Training setup ready for Kaggle/Colab execution.')
        return

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
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
