import os, sys, json, pandas as pd, numpy as np, tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

split_df = pd.read_csv('splits/split.csv')
train_df = split_df[split_df['strat_fold'].isin(range(1, 9))].copy()
val_df = split_df[split_df['strat_fold'] == 9].copy()
test_df = split_df[split_df['strat_fold'] == 10].copy()

labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
label_to_id = {l: i for i, l in enumerate(labels)}

train_df['class_id'] = train_df['label'].map(label_to_id)
val_df['class_id'] = val_df['label'].map(label_to_id)
test_df['class_id'] = test_df['label'].map(label_to_id)

Model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
pbase = layers.GlobalAveragePooling2D()(Model.output)
pbase = layers.Dropout(0.3)(pbase)
o = layers.Dense(3, activation='softmax')(pbase)
model = models.Model(inputs=Model.input, outputs=o)
model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
print('Model built successfully')
