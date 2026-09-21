import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import ast
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
import wfdb

# Fixed Seed for Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print('TF version:', tf.__version__)

os.makedirs('results', exist_ok=True)
os.makedirs('app/samples', exist_ok=True)
os.makedirs('model', exist_ok=True)

db_path = 'ptbxl_database.csv'
scp_path = 'scp_statements.csv'

df_db = pd.read_csv(db_path, index_col='ecg_id')
df_scp = pd.read_csv(scp_path, index_col=0)

df_db['scp_codes'] = df_db['scp_codes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def assign_label(scp_dict):
    if not isinstance(scp_dict, dict) or len(scp_dict) == 0:
        return None
    valid_classes = set()
    for code, likelihood in scp_dict.items():
        if likelihood >= 50.0 and code in df_scp.index:
            sc = df_scp.loc[code, 'diagnostic_class']
            if pd.notna(sc) and str(sc) != '':
                valid_classes.add(str(sc))
    if not valid_classes:
        return None
    if 'MI' in valid_classes:
        return 'MI'
    if valid_classes == {'NORM'}:
        return 'NORM'
    return 'OTHER_ABNORMAL'

df_db['label'] = df_db['scp_codes'].apply(assign_label)
excluded_count = df_db['label'].isna().sum()
clean_df = df_db[df_db['label'].notna()].copy()

print('Total PTB-XL records:', len(df_db))
print('Excluded records (no usable code):', excluded_count)
print('Usable labeled records:', len(clean_df))

labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
label_to_id = {l: i for i, l in enumerate(labels)}

X_list, y_list, folds_list, ids_list = [], [], [], []
sample_records = []

print('Loading WFDB 100Hz signals...')
start_time = time.time()

for ecg_id, row in clean_df.iterrows():
    rel = str(row['filename_lr']).replace('\\', '/')
    fp1 = os.path.normpath(os.path.join('data/ptbxl_100hz', rel))
    fp2 = os.path.normpath(rel)
    fp = fp1 if os.path.exists(fp1 + '.dat') else fp2
    if not (os.path.exists(fp + '.hea') and os.path.exists(fp + '.dat')):
        continue
            
    try:
        record = wfdb.rdrecord(fp)
        signal = record.p_signal
        if signal.shape != (1000, 12):
            continue
        signal_mean = np.mean(signal, axis=0, keepdims=True)
        signal_std = np.std(signal, axis=0, keepdims=True) + 1e-6
        signal_norm = (signal - signal_mean) / signal_std
        
        X_list.append(signal_norm)
        y_list.append(label_to_id[row['label']])
        folds_list.append(row['strat_fold'])
        ids_list.append(ecg_id)
        
        if len(sample_records) < 12:
            lbl = row['label']
            lbl_count = sum(1 for s in sample_records if s['label'] == lbl)
            if lbl_count < 4:
                sample_file = f'app/samples/sample_{ecg_id}_{lbl}.npy'
                np.save(sample_file, signal)
                sample_records.append({
                    'id': int(ecg_id),
                    'patient_id': int(row['patient_id']),
                    'age': int(row['age']) if pd.notna(row['age']) else 60,
                    'sex': 0 if row['sex'] == 0 else 1,
                    'label': lbl,
                    'file': sample_file
                })
    except Exception:
        continue

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
folds = np.array(folds_list, dtype=np.int32)

print('Loaded', len(X), 'records in', round(time.time() - start_time, 1), 's. Shape:', X.shape)

train_mask = np.isin(folds, range(1, 9))
val_mask = (folds == 9)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]

print('Split counts -> Train (folds 1-8):', len(X_train), 'Val (fold 9):', len(X_val))

cw_vec = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=y_train)
class_weights = dict(zip([0, 1, 2], cw_vec))
print('Class weights:', class_weights)

with open('app/samples/samples_index.json', 'w') as f:
    json.dump(sample_records, f, indent=2)
print('Saved', len(sample_records), 'sample records to app/samples/samples_index.json')

def build_1d_cnn(input_shape=(1000, 12), num_classes=3):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(32, kernel_size=7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model_1d = build_1d_cnn()

cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

print('Training Waveform 1D CNN Model on full dataset...')
history = model_1d.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=64,
    class_weight=class_weights,
    callbacks=cb,
    verbose=1
)

model_1d.save('model/waveform_1d_cnn.h5')
print('Saved model/waveform_1d_cnn.h5')
