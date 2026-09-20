# Model Card: CardioScan 1D Waveform CNN

## 1. Model Details
- **Model Name**: CardioScan 1D Waveform CNN (`model/waveform_1d_cnn.h5`)
- **Model Architecture**: 1D Convolutional Neural Network (Conv1D + Batch Normalization + MaxPool1D + GlobalAveragePooling1D + Dense)
- **Model Version**: 1.0.0
- **Input Format**: 12-lead raw ECG waveform signals sampled at 100 Hz (array shape `(1000, 12)`), standardized via per-channel z-score `(x - mean) / (std + 1e-6)`.
- **Output Classes**: `NORM` (Normal Electrocardiogram), `MI` (Myocardial Infarction), `OTHER_ABNORMAL` (STTC / CD / HYP).

---

## 2. Intended Use
- **Intended Purpose**: Educational demonstration, academic research, and benchmarking of 1D deep learning architectures on raw digital ECG signals.
- **Target Audience**: Students, researchers, and developers exploring machine learning for biomedical signal processing.
- **Primary Use Case**: Interactive demo displaying 12-lead ECG plots alongside model diagnostic class probabilities and rule-based heart rate analysis.

---

## 3. Data & Labeling Rules
- **Primary Dataset**: PTB-XL Electrocardiography Dataset v1.0.3 (Wagner et al., 2020), available via PhysioNet.
- **Diagnostic Superclass Ground Truth Rules**:
  - Derived from official SCP statement definitions (`scp_statements.csv`) with `likelihood >= 50.0`.
  - **Priority 1 (`MI`)**: Assigned if Myocardial Infarction (`MI`) is present in diagnostic superclasses.
  - **Priority 2 (`NORM`)**: Assigned if Normal (`NORM`) is the sole diagnostic superclass.
  - **Priority 3 (`OTHER_ABNORMAL`)**: Assigned if any other superclass (`STTC`, `CD`, `HYP`) is present.
  - **Excluded**: Records with no usable diagnostic codes with likelihood $\ge 50$ (1,426 records excluded).
- **Label Agreement**: Comparing legacy folder labels with PTB-XL ground truth revealed a 77.2% agreement rate (22.8% discrepancy resolved).

---

## 4. Evaluation & Leak-Free Split
- **Split Strategy**: Official PTB-XL patient-wise `strat_fold` split:
  - **Train Set (Folds 1–8)**: 15,958 records
  - **Validation Set (Fold 9)**: 2,183 records (used solely for hyperparameter tuning)
  - **Test Set (Fold 10)**: 2,050 records (evaluated ONCE)

### Benchmark Test Results (Full Fold 10, N=2,050)

| Model / Baseline Strategy | Test Samples ($N$) | Test Accuracy (95% CI) | Macro F1 (95% CI) | Macro AUROC (95% CI) | Status vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline ("Always Predict NORM")** | 2,050 | 0.5565 [0.4937, 0.6192] | 0.2384 [0.2204, 0.2547] | 0.5000 [0.5000, 0.5000] | Baseline |
| **Random Class Frequency Baseline** | 2,050 | 0.4100 [0.3515, 0.4728] | 0.3359 [0.2801, 0.3900] | 0.5000 [0.4350, 0.5650] | Baseline |
| **2D Image Model (EfficientNetB0)** | 239 | 0.3598 [0.3013, 0.4226] | 0.3188 [0.2578, 0.3770] | 0.5103 [0.4412, 0.5794] | Failed (Did not beat baseline; cause not established) |
| **1D Waveform Model (1D CNN)** | 2,050 | **0.7051** [0.6282, 0.7756] | **0.6272** [0.5379, 0.7060] | **0.8532** [0.8028, 0.9002] | **PASSED (Beats baseline; 95% CI no overlap)** |

---

## 5. Known Limitations
1. **Educational Only**: Not a certified medical device and not clinically validated for patient diagnosis or triage.
2. **Signal Resolution**: Evaluated on 100 Hz downsampled signals (`filename_lr`); higher frequency components (e.g. 500 Hz) are not utilized.
3. **Imbalanced Test Subclasses**: Certain specific subclass conditions within `OTHER_ABNORMAL` have lower representation in Fold 10.
4. **2D Image Model Non-Generalization**: 2D plot image models trained on visual renders failed to generalize across patient splits (Macro F1 0.3188); 2D image uploads in the web application do NOT produce neural network predictions and display rule-based signal metrics only.

---

## 6. Out-of-Scope Uses
- **Clinical Diagnosis & Treatment**: Must NOT be used for real-time patient monitoring, emergency diagnostic triage, or replacement of physician interpretation.
- **Single-Lead Consumer ECGs**: Not designed or trained for 1-lead smart watch or wearable consumer ECG signals.
