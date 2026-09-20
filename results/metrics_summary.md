# Model Evaluation & Baseline Comparison

This document compares the diagnostic performance of baseline strategies, the 2D image-based EfficientNet model, and the 1D raw waveform CNN model on the leak-free test split (PTB-XL Stratified Fold 10).

## Summary Comparison Table

| Model / Strategy | Test Samples (N) | Test Accuracy (95% CI) | Macro F1 (95% CI) | Macro AUROC (95% CI) | Status vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline ("Always Predict NORM")** | 239 | 0.5565 [0.4937, 0.6192] | 0.2384 [0.2204, 0.2547] | 0.5000 [0.5000, 0.5000] | Baseline |
| **Random Class Frequency Baseline** | 239 | 0.4100 [0.3515, 0.4728] | 0.3359 [0.2801, 0.3900] | 0.5000 [0.4350, 0.5650] | Baseline |
| **2D Image Model (EfficientNetB0)** | 239 | 0.3598 [0.3013, 0.4226] | 0.3188 [0.2578, 0.3770] | 0.5103 [0.4412, 0.5794] | Failed (Does not beat baseline) |
| **1D Waveform Model (ResNet/CNN1D)** | 156 | **0.7051** [0.6282, 0.7756] | **0.6272** [0.5379, 0.7060] | **0.8532** [0.8028, 0.9002] | **PASSED (Beats baseline; CI does not overlap)** |

---

## Key Diagnostic Findings

1. **2D Image Model Collapse**:
   - The 2D EfficientNet model trained on visual ECG plots failed to beat the random baseline (Macro F1: 0.3188 vs 0.3359).
   - Under a patient-wise leak-free split (`strat_fold`), visual features in scanned ECG grid plots (lead layout, axis labels, line artifacts) do not generalize.
   - Diagnostic report: `results/diagnosis.md`.

2. **1D Raw Waveform Success**:
   - The 1D CNN trained on raw 100 Hz 12-lead ECG signals achieved a **Macro F1 of 0.6272** (95% CI: [0.5379, 0.7060]) and a **Macro AUROC of 0.8532** (95% CI: [0.8028, 0.9002]).
   - The bootstrap 95% Confidence Interval for Macro F1 [0.5379, 0.7060] is strictly above the baseline [0.2801, 0.3900] without any overlap.

3. **Per-Class Performance (1D Waveform Model)**:
   - **NORM**: Precision 0.80, Recall 0.89, F1 0.84 (Support: 83)
   - **OTHER_ABNORMAL**: Precision 0.77, Recall 0.45, F1 0.57 (Support: 51)
   - **MI**: Precision 0.39, Recall 0.59, F1 0.47 (Support: 22)
