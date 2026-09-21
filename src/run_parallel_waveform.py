import os
import sys
import json
import ast
import urllib.request
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from concurrent.futures import ThreadPoolExecutor
import wfdb

print('TF version:', tf.__version__)

DATA_DIR = 'data/ptbxl_100hz'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('app/samples', exist_ok=True)
os.makedirs('model', exist_ok=True)

db_path = 'ptbxl_database.csv'
scp_path = 'scp_statements.csv'

if not os.path.exists(db_path):
    print('Downloading ptbxl_database.csv...')
    urllib.request.urlretrieve('https://physionet.org/content/ptb-xl/1.0.3/ptbxl_database.csv', db_path)

if not os.path.exists(scp_path):
    print('Downloading scp_statements.csv...')
    urllib.request.urlretrieve('https://physionet.org/content/ptb-xl/1.0.3/scp_statements.csv', scp_path)

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
            if pd.notna(sc) and sc != '':
                valid_classes.add(str(sc))
    if not valid_classes:
        return None
    if 'MI' in valid_classes:
        return 'MI'
    if valid_classes == {'NORM'}:
        return 'NORM'
    return 'OTHER_ABNORMAL'

df_db['label'] = df_db['scp_codes'].apply(assign_label)
valid_mask = df_db['label'].notna()
excluded_count = len(df_db) - valid_mask.sum()
print('Total PTB-XL records:', len(df_db), 'Excluded without usable code:', excluded_count, 'Valid:', valid_mask.sum())

clean_df = df_db[valid_mask].copy()

BASE_URL = 'https://physionet.org/content/ptb-xl/1.0.3/'

def download_record_files(fname):
    dat_rel = fname + '.dat'
    hea_rel = fname + '.hea'
    dat_out = os.path.join(DATA_DIR, dat_rel)
    hea_out = os.path.join(DATA_DIR, hea_rel)
    os.makedirs(os.path.dirname(dat_out), exist_ok=True)
    if not os.path.exists(dat_out):
        try:
            urllib.request.urlretrieve(BASE_URL + dat_rel, dat_out)
        except Exception:
            pass
    if not os.path.exists(hea_out):
        try:
            urllib.request.urlretrieve(BASE_URL + hea_rel, hea_out)
        except Exception:
            pass

records_to_download = clean_df['filename_lr'].unique().tolist()
print('Downloading record files in parallel (ThreadPoolExecutor, 20 workers)...')
with ThreadPoolExecutor(max_workers=20) as executor:
    list(executor.map(download_record_files, records_to_download[:3000]))

print('Parallel download completed!')

labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
label_to_id = {l: i for i, l in enumerate(labels)}

X_list, y_list, folds_list, ids_list = [], [], [], []
sample_saved_count = 0

for ecg_id, row in clean_df.iterrows():
    rec_path = os.path.join(DATA_DIR, row['filename_lr'])
    if not os.path.exists(rec_path + '.dat'):
        continue
    try:
        record = wfdb.rdrecord(rec_path)
        signal = record.p_signal # shape (1000, 12)
        if signal.shape != (1000, 12):
            continue
        X_list.append(signal)
        y_list.append(label_to_id[row['label']])
        folds_list.append(row['strat_fold'])
        ids_list.append(ecg_id)
        
        # Save sample files for app/samples (up to 10 records)
        if sample_saved_count < 10 and row['strat_fold'] == 10:
            sample_file = 'app/samples/sample_' + str(ecg_id) + '_' + str(row['label']) + '.npy'
            np.save(sample_file, signal)
            sample_saved_count += 1
    except Exception as e:
        continue

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)
folds = np.array(folds_list, dtype=np.int32)

print('Loaded records:', len(X), 'X shape:', X.shape, 'y shape:', y.shape)

train_mask = np.isin(folds, range(1, 9))
val_mask = (folds == 9)
test_mask = (folds == 10)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print('Train:', len(X_train), 'Val:', len(X_val), 'Test:', len(X_test))

cw_vec = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=y_train)
class_weights = dict(zip([0, 1, 2], cw_vec))
print('Class weights:', class_weights)

def build_1d_cnn(input_shape=(1000, 12), num_classes=3):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=7, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(64, kernel_size=5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print('Training 1D CNN Waveform Model...')
model_1d = build_1d_cnn()

history = model_1d.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)

model_1d.save('model/waveform_1d_cnn.h5')
print('Saved model/waveform_1d_cnn.h5')

# Evaluate ONCE on Test Fold (Fold 10)
print('Waveform 1D CNN Test Fold Evaluation (Fold 10)')
test_preds_prob = model_1d.predict(X_test)
test_preds = np.argmax(test_preds_prob, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_macro_f1 = f1_score(y_test, test_preds, average='macro')
y_test_cat = tf.keras.utils.to_categorical(y_test, 3)
try:
    test_auroc = roc_auc_score(y_test_cat, test_preds_prob, multi_class='ovr', average='macro')
except Exception as e:
    test_auroc = 0.5

print('Waveform Test Accuracy:', test_acc)
print('Waveform Test Macro F1:', test_macro_f1)
print('Waveform Test Macro AUROC:', test_auroc)
cm = confusion_matrix(y_test, test_preds)
print('Confusion Matrix:\n', cm)
rep = classification_report(y_test, test_preds, target_names=labels)
print('Classification Report:\n', rep)

np.random.seed(42)
n_bootstraps = 1000
boot_f1s, boot_accs, boot_aurocs = [], [], []
n_samples = len(y_test)

for _ in range(n_bootstraps):
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    if len(np.unique(y_test[indices])) < 3:
        continue
    boot_accs.append(accuracy_score(y_test[indices], test_preds[indices]))
    boot_f1s.append(f1_score(y_test[indices], test_preds[indices], average='macro'))
    try:
        auroc = roc_auc_score(y_test_cat[indices], test_preds_prob[indices], multi_class='ovr', average='macro')
        boot_aurocs.append(auroc)
    except:
        pass

f1_ci = np.percentile(boot_f1s, [2.5, 97.5])
acc_ci = np.percentile(boot_accs, [2.5, 97.5])
auroc_ci = np.percentile(boot_aurocs, [2.5, 97.5])

print('Macro F1 95% CI:', f1_ci)
print('Accuracy 95% CI:', acc_ci)
print('Macro AUROC 95% CI:', auroc_ci)

results_summary = {
    'waveform_1d_cnn': {
        'test_samples': int(len(y_test)),
        'accuracy': float(test_acc),
        'accuracy_ci': [float(acc_ci[0]), float(acc_ci[1])],
        'macro_f1': float(test_macro_f1),
        'macro_f1_ci': [float(f1_ci[0]), float(f1_ci[1])],
        'macro_auroc': float(test_auroc),
        'macro_auroc_ci': [float(auroc_ci[0]), float(auroc_ci[1])],
        'confusion_matrix': cm.tolist()
    }
}

with open('results/metrics_waveform.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print('Waveform evaluation completed and saved to results/metrics_waveform.json')
