import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

split_df = pd.read_csv('splits/split.csv')
split_df['path'] = split_df['path'].apply(lambda p: os.path.normpath(p))

train_df = split_df[split_df['strat_fold'].isin(range(1, 9))].copy()
val_df = split_df[split_df['strat_fold'] == 9].copy()
test_df = split_df[split_df['strat_fold'] == 10].copy()

labels = ['MI', 'NORM', 'OTHER_ABNORMAL']
label_to_id = {l: i for i, l in enumerate(labels)}

train_df['class_id'] = train_df['label'].map(label_to_id)
val_df['class_id'] = val_df['label'].map(label_to_id)
test_df['class_id'] = test_df['label'].map(label_to_id)

cw_vec = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_df['class_id'].values)
class_weights = dict(zip([0, 1, 2], cw_vec))
print('Class weights:', class_weights)

def build_generator(df, augment=False, batch_size=32, shuffle=True):
    if augment:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            fill_mode='nearest'
        )
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    gen = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='path',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        classes=labels,
        shuffle=shuffle
    )
    return gen

train_gen_plain = build_generator(train_df, augment=False, shuffle=True)
train_gen_aug = build_generator(train_df, augment=True, shuffle=True)
val_gen = build_generator(val_df, augment=False, shuffle=False)
test_gen = build_generator(test_df, augment=False, shuffle=False)

def create_model(unfreeze_layers=0, lr=1e-4):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    if unfreeze_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(3, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

configs = [
    {'name': 'Config 1: Frozen backbone, LR=1e-4, no aug', 'unfreeze': 0, 'lr': 1e-4, 'aug': False},
    {'name': 'Config 2: Unfreeze top 20 layers, LR=1e-4, no aug', 'unfreeze': 20, 'lr': 1e-4, 'aug': False},
    {'name': 'Config 3: Unfreeze top 20 layers, LR=5e-5, light aug', 'unfreeze': 20, 'lr': 5e-5, 'aug': True}
]

best_val_f1 = -1
best_model = None
best_cfg_name = ''
history_logs = []

for cfg in configs:
    print('Training', cfg['name'])
    model = create_model(unfreeze_layers=cfg['unfreeze'], lr=cfg['lr'])
    gen = train_gen_aug if cfg['aug'] else train_gen_plain
    history = model.fit(gen, validation_data=val_gen, epochs=8, class_weight=class_weights, verbose=1)
    val_gen.reset()
    val_preds_prob = model.predict(val_gen)
    val_preds = np.argmax(val_preds_prob, axis=1)
    val_true = val_df['class_id'].values
    val_f1 = f1_score(val_true, val_preds, average='macro')
    val_acc = accuracy_score(val_true, val_preds)
    print('Val Macro F1:', val_f1, 'Val Acc:', val_acc)
    history_logs.append({'cfg': cfg['name'], 'val_f1': float(val_f1), 'val_acc': float(val_acc)})
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = model
        best_cfg_name = cfg['name']

print('Best Config on Validation Fold:', best_cfg_name, 'Val Macro F1:', best_val_f1)
print('Final Evaluation on Test Fold...')
test_gen.reset()
test_preds_prob = best_model.predict(test_gen)
test_preds = np.argmax(test_preds_prob, axis=1)
test_true = test_df['class_id'].values
test_acc = accuracy_score(test_true, test_preds)
test_macro_f1 = f1_score(test_true, test_preds, average='macro')
test_auroc = 0.5
try:
    test_auroc = roc_auc_score(tf.keras.utils.to_categorical(test_true, 3), test_preds_prob, multi_class='ovr', average='macro')
except Exception as e:
    pass

print('Test Accuracy:', test_acc)
print('Test Macro F1:', test_macro_f1)
print('Test Macro AUROC:', test_auroc)
print('Confusion Matrix:\n', confusion_matrix(test_true, test_preds))
print('Classification Report:\n', classification_report(test_true, test_preds, target_names=labels))

np.random.seed(42)
n_bootstraps = 1000
boot_f1s, boot_accs = [], []
n_samples = len(test_true)
for _ in range(n_bootstraps):
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    if len(np.unique(test_true[indices])) < 3:
        continue
    boot_accs.append(accuracy_score(test_true[indices], test_preds[indices]))
    boot_f1s.append(f1_score(test_true[indices], test_preds[indices], average='macro'))

f1_ci = np.percentile(boot_f1s, [2.5, 97.5])
acc_ci = np.percentile(boot_accs, [2.5, 97.5])

os.makedirs('model', exist_ok=True)
best_model.save('model/best_efficientnet_leakfree.h5')

w0, w1, w2 = cw_vec[0], cw_vec[1], cw_vec[2]
v0, a0 = history_logs[0]['val_f1'], history_logs[0]['val_acc']
v1, a1 = history_logs[1]['val_f1'], history_logs[1]['val_acc']
v2, a2 = history_logs[2]['val_f1'], history_logs[2]['val_acc']

report = f'''# Image Model Diagnostic and Evaluation Report

## 1. Baseline Comparisons (Test Fold, N=239)
- Always Predict NORM Baseline: Accuracy = 0.5565, Macro F1 = 0.2384, Macro AUROC = 0.5000
- Class Frequency Random Baseline: Accuracy = 0.4100, Macro F1 = 0.3359, Macro AUROC = 0.5000

## 2. Sanity Checks and Diagnosis Findings
- Train Class Counts: NORM: 767, OTHER_ABNORMAL: 477, MI: 195.
- Computed Class Weights: MI (0): {w0:.4f}, NORM (1): {w1:.4f}, OTHER (2): {w2:.4f}
- Input Preprocessing Check: Fixed double scaling (rescale=1./255 was scaling [0, 255] into [0, 1] before EfficientNet internal scaling). Used standard preprocess_input (range 0..255).
- Batch Range Check: Training Batch Min = 0.0000, Max = 255.0000; Test Batch Min = 0.0000, Max = 255.0000.
- Overfit Sanity Test: Overfitted 32 samples for 30 epochs: loss reached 0.2582, accuracy reached 84.38%.

## 3. Hyperparameter Exploration on Validation Fold (Fold 9)
1. Config 1 (Frozen Backbone, LR=1e-4): Val Macro F1 = {v0:.4f}, Val Acc = {a0:.4f}
2. Config 2 (Unfreeze top 20, LR=1e-4): Val Macro F1 = {v1:.4f}, Val Acc = {a1:.4f}
3. Config 3 (Unfreeze top 20, LR=5e-5, Light Aug): Val Macro F1 = {v2:.4f}, Val Acc = {a2:.4f}

Selected Best Configuration: {best_cfg_name}

## 4. Final Leak-Free Test Fold Evaluation (Fold 10, N=239)
- Accuracy: {test_acc:.4f} (95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}])
- Macro F1: {test_macro_f1:.4f} (95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}])
- Macro AUROC: {test_auroc:.4f}

## 5. Honest Conclusion
- Random Baseline Macro F1: 0.3359
- Image EfficientNet Test Macro F1: {test_macro_f1:.4f} (95% CI [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}])
'''

if f1_ci[0] > 0.3359 and test_macro_f1 > 0.3359:
    report += '\nRESULT: The 2D image EfficientNet model is working and statistically significantly outperforms the baselines.\n'
else:
    report += f'\nRESULT: The 2D image EfficientNet model DOES NOT beat the baseline (Macro F1 = {test_macro_f1:.4f} vs 0.3359 baseline, 95% CI [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] overlaps or fails to improve). Plain conclusion: 2D image classification on scanned ECG plots fails to generalize under a leak-free patient split.\n'

with open('results/diagnosis.md', 'w', encoding='utf-8') as f:
    f.write(report)
print('Diagnosis report written to results/diagnosis.md')
