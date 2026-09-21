# Model Card: CardioScan 1D Waveform CNN

## 1. Model Details
- **Model Name**: CardioScan 1D Waveform CNN (model/waveform_1d_cnn.h5)
- **Model Architecture**: 1D Convolutional Neural Network (Conv1D + Batch Normalization + MaxPool1D + GlobalAveragePooling1D + Dense)
- **Model Version**: 2.0.0 (Full Dataset Retrained)
- **Input Format**: 12-lead raw ECG waveform signals sampled at 100 Hz (array shape (1000, 12)), standardized via per-channel z-score (x - mean) / (std + 1e-6).
- **Output Classes**: NORM (Normal Electrocardiogram), MI (Myocardial Infarction), OTHER_ABNORMAL (STTC / CD / HYP).

---

## 2. Intended Use
- **Intended Purpose**: Educational demonstration, academic research, and benchmarking of 1D deep learning architectures on raw digital ECG signals.
- **Target Audience**: Students, researchers, and developers exploring machine learning for biomedical signal processing.
- **Primary Use Case**: Interactive web app displaying 12-lead ECG plots alongside model diagnostic class probabilities and rule-based signal metrics.

---

## 3. Data & Labeling Rules
- **Primary Dataset**: PTB-XL Electrocardiography Dataset v1.0.3 (Wagner et al., 2020), available via PhysioNet.
- **Diagnostic Superclass Ground Truth Rules**:
  - Derived from official SCP statement definitions (scp_statements.csv) with likelihood >= 50.0.
  - **Priority 1 (MI)**: Assigned if Myocardial Infarction (MI) is present in diagnostic superclasses.
  - **Priority 2 (NORM)**: Assigned if Normal (NORM) is sole diagnostic superclass.
  - **Priority 3 (OTHER_ABNORMAL)**: Assigned if any other superclass (STTC, CD, HYP) is present.
- **Training Coverage**: Folds 1-8 (16,303 records) for training, Fold 9 (2,034 records) for validation tuning, Fold 10 (2,050 records) for single test fold evaluation.

---

## 4. Evaluation & Leak-Free Split Results

Benchmark Test Results (PTB-XL Stratified Fold 10, N=2,050):

| Model / Baseline Strategy | Test Samples (N) | Test Accuracy (95% CI) | Macro F1 (95% CI) | Macro AUROC (95% CI) | MI Recall | NORM Recall | OTHER Recall | Status vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline (Always Predict NORM)** | 2,050 | 0.4434 [0.4215, 0.4629] | 0.2048 [0.1977, 0.2110] | 0.5000 [0.5000, 0.5000] | 0.0000 (0/415) | 1.0000 (909/909) | 0.0000 (0/726) | Baseline |
| **Random Class Frequency Baseline** | 2,050 | 0.3624 [0.3405, 0.3844] | 0.3321 [0.3122, 0.3534] | 0.5000 [0.5000, 0.5000] | 0.2458 (102/415) | 0.4851 (441/909) | 0.2879 (209/726) | Baseline |
| **2D Image Model (EfficientNetB0)** | 239* | 0.5565 [0.4937, 0.6192] | 0.2384 [0.2201, 0.2547] | 0.5061 [0.4610, 0.5512] | 0.0000 (0/42) | 1.0000 (133/133) | 0.0000 (0/64) | Majority Collapse |
| **1D Waveform Model (1D CNN)** | 2,050 | **0.7644** [0.7449, 0.7824] | **0.7383** [0.7175, 0.7582] | **0.9005** [0.8891, 0.9104] | **0.6289** (261/415) | **0.9153** (832/909) | **0.6529** (474/726) | **PASSED (Beats baseline)** |

*Note: 2D image model evaluated on image split (N=239).

---

## 5. Known Limitations
1. **Educational Only**: Not a certified medical device and not clinically validated for patient diagnosis or triage.
2. **Signal Resolution**: Evaluated on 100 Hz downsampled signals (ilename_lr).
3. **2D Image Model Non-Generalization**: 2D plot image models trained on visual renders collapsed to majority class predictions under leak-free patient splits.
