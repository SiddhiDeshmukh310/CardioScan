import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
import ast
import time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import wfdb

# Fixed Seed
SEED = 42
np.random.seed(SEED)

print('Evaluating Waveform 1D CNN Model...')
MODEL_PATH = 'model/waveform_1d_cnn.h5'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file {MODEL_PATH} not found!")

model_1d = tf.keras.models.load_model(MODEL_PATH)
print("Loaded 1D Waveform CNN model successfully.")

os.makedirs('results', exist_ok=True)

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
clean_df = df_db[df_db['label'].notna()].copy()

labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
label_to_id = {l: i for i, l in enumerate(labels)}

test_df = clean_df[clean_df['strat_fold'] == 10].copy()
print(f"Full Test Fold (Fold 10) total labeled records: {len(test_df)}")

# Pre-allocated arrays to optimize memory
y_test_list = []
preds_prob_list = []

for ecg_id, row in test_df.iterrows():
    rel = str(row['filename_lr']).replace(chr(92), '/')
    fp = os.path.normpath(os.path.join('data/ptbxl_100hz', rel))
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
        
        inp = np.expand_dims(signal_norm.astype(np.float32), axis=0)
        prob = model_1d.predict(inp, verbose=0)[0]
        
        preds_prob_list.append(prob)
        y_test_list.append(label_to_id[row['label']])
    except Exception:
        continue

y_test = np.array(y_test_list, dtype=np.int32)
test_preds_prob = np.array(preds_prob_list, dtype=np.float32)
N_test = len(y_test)
print(f"Loaded and evaluated {N_test} valid test records for Fold 10.")

test_preds = np.argmax(test_preds_prob, axis=1)

test_acc = accuracy_score(y_test, test_preds)
test_macro_f1 = f1_score(y_test, test_preds, average='macro')
y_test_cat = tf.keras.utils.to_categorical(y_test, 3)

try:
    test_auroc = roc_auc_score(y_test_cat, test_preds_prob, multi_class='ovr', average='macro')
except Exception:
    test_auroc = 0.5

majority_pred = np.ones_like(y_test)
maj_acc = accuracy_score(y_test, majority_pred)
maj_f1 = f1_score(y_test, majority_pred, average='macro')

train_clean = clean_df[clean_df['strat_fold'].isin(range(1, 9))]
class_counts = train_clean['label'].value_counts()
class_probs = [class_counts['MI'] / len(train_clean), class_counts['NORM'] / len(train_clean), class_counts['OTHER_ABNORMAL'] / len(train_clean)]
np.random.seed(42)
random_preds = np.random.choice([0, 1, 2], size=N_test, p=class_probs)
rand_acc = accuracy_score(y_test, random_preds)
rand_f1 = f1_score(y_test, random_preds, average='macro')

print(f"=== FULL TEST FOLD (N={N_test}) EVALUATION RESULTS ===")
print(f"1D Waveform CNN Accuracy: {test_acc:.4f}, Macro F1: {test_macro_f1:.4f}, Macro AUROC: {test_auroc:.4f}")
print(f"Majority Baseline Accuracy: {maj_acc:.4f}, Macro F1: {maj_f1:.4f}")
print(f"Random Baseline Accuracy: {rand_acc:.4f}, Macro F1: {rand_f1:.4f}")

np.random.seed(42)
n_bootstraps = 1000
boot_f1s, boot_accs, boot_aurocs = [], [], []

for _ in range(n_bootstraps):
    indices = np.random.choice(N_test, size=N_test, replace=True)
    if len(np.unique(y_test[indices])) < 3:
        continue
    boot_accs.append(accuracy_score(y_test[indices], test_preds[indices]))
    boot_f1s.append(f1_score(y_test[indices], test_preds[indices], average='macro'))
    try:
        auroc = roc_auc_score(y_test_cat[indices], test_preds_prob[indices], multi_class='ovr', average='macro')
        boot_aurocs.append(auroc)
    except Exception:
        pass

f1_ci = np.percentile(boot_f1s, [2.5, 97.5])
acc_ci = np.percentile(boot_accs, [2.5, 97.5])
auroc_ci = np.percentile(boot_aurocs, [2.5, 97.5])

boot_maj_f1s, boot_maj_accs = [], []
boot_rand_f1s, boot_rand_accs = [], []

for _ in range(n_bootstraps):
    indices = np.random.choice(N_test, size=N_test, replace=True)
    if len(np.unique(y_test[indices])) < 3:
        continue
    boot_maj_accs.append(accuracy_score(y_test[indices], majority_pred[indices]))
    boot_maj_f1s.append(f1_score(y_test[indices], majority_pred[indices], average='macro'))
    boot_rand_accs.append(accuracy_score(y_test[indices], random_preds[indices]))
    boot_rand_f1s.append(f1_score(y_test[indices], random_preds[indices], average='macro'))

maj_acc_ci = np.percentile(boot_maj_accs, [2.5, 97.5])
maj_f1_ci = np.percentile(boot_maj_f1s, [2.5, 97.5])
rand_acc_ci = np.percentile(boot_rand_accs, [2.5, 97.5])
rand_f1_ci = np.percentile(boot_rand_f1s, [2.5, 97.5])

cm = confusion_matrix(y_test, test_preds)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
plt.title(f'1D Waveform CNN Confusion Matrix (N={N_test})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('results/confusion_matrix_waveform.png', dpi=150)
plt.close()

rep = classification_report(y_test, test_preds, target_names=labels, output_dict=True)
print("Classification Report:\n", classification_report(y_test, test_preds, target_names=labels))

results_summary = {
    'test_samples': int(N_test),
    'waveform_1d_cnn': {
        'test_samples': int(N_test),
        'accuracy': float(test_acc),
        'accuracy_ci': [float(acc_ci[0]), float(acc_ci[1])],
        'macro_f1': float(test_macro_f1),
        'macro_f1_ci': [float(f1_ci[0]), float(f1_ci[1])],
        'macro_auroc': float(test_auroc),
        'macro_auroc_ci': [float(auroc_ci[0]), float(auroc_ci[1])],
        'confusion_matrix': cm.tolist(),
        'per_class_report': rep
    },
    'majority_baseline': {
        'test_samples': int(N_test),
        'accuracy': float(maj_acc),
        'accuracy_ci': [float(maj_acc_ci[0]), float(maj_acc_ci[1])],
        'macro_f1': float(maj_f1),
        'macro_f1_ci': [float(maj_f1_ci[0]), float(maj_f1_ci[1])],
        'macro_auroc': 0.5
    },
    'random_baseline': {
        'test_samples': int(N_test),
        'accuracy': float(rand_acc),
        'accuracy_ci': [float(rand_acc_ci[0]), float(rand_acc_ci[1])],
        'macro_f1': float(rand_f1),
        'macro_f1_ci': [float(rand_f1_ci[0]), float(rand_f1_ci[1])],
        'macro_auroc': 0.5
    },
    'image_2d_model': {
        'test_samples': 239,
        'accuracy': 0.3598,
        'accuracy_ci': [0.3013, 0.4226],
        'macro_f1': 0.3188,
        'macro_f1_ci': [0.2578, 0.3770],
        'macro_auroc': 0.5103,
        'note': 'Evaluated on subset of 239 images; did not beat baseline; cause not established'
    }
}

with open('results/metrics_waveform.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Saved results/metrics_waveform.json and results/confusion_matrix_waveform.png")
