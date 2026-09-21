# Image Model Diagnostic and Evaluation Report

## 1. Baseline Comparisons (Test Fold, N=239)
- Always Predict NORM Baseline: Accuracy = 0.5565, Macro F1 = 0.2384, Macro AUROC = 0.5000
- Class Frequency Random Baseline: Accuracy = 0.4100, Macro F1 = 0.3359, Macro AUROC = 0.5000

## 2. Sanity Checks and Diagnosis Findings
- Train Class Counts: NORM: 767, OTHER_ABNORMAL: 477, MI: 195.
 - Computed Class Weights: MI (0): 2.4598, NORM (1): 0.6254, OTHER (2): 1.0056
- Input Preprocessing Check: Fixed double scaling (rescale=1/255 was scaling [0, 255] into [0, 1] before EfficientNet internal scaling). Used standard preprocess_input.
- Batch Range Check: Training Batch Min = 0.0, Max = 255.0; Test Batch Min = 0.0, Max = 255.0.
- Overfit Sanity Test: Overfitted 32 samples for 30 epochs: loss reached 0.2582, accuracy reached 84.38%.

## 3. Hyperparameter Exploration on Validation Fold (Fold 9)
1. Config 1 (LR=1e-4, Dropout 0.3): Val Macro F1 = 0.3714, Val Acc = 0.3921
2. Config 2 (LR=5e-4, Dropout 0.5): Val Macro F1 = 0.2822, Val Acc = 0.3084
3. Config 3 (LR=1e-3, No Class Weights): Val Macro F1 = 0.2354, Val Acc = 0.4493

Selected Best Configuration: Config 1: LR=1e-4, Dropout 0.3, Class Weights

## 4. Final Leak-Free Test Fold Evaluation (Fold 10, N=239)
- Accuracy: 0.3598 (95% CI: [0.3013, 0.4226])
- Macro F1: 0.3188 (95% CI: [0.2578, 0.377])
- Macro AUROC: 0.5103

## 5. Honest Conclusion
- Random Baseline Macro F1: 0.3359
- Image EfficientNet Test Macro F1: 0.3188 (95% CI [0.2578, 0.377])

**RESULT**: The 2D image EfficientNet model DOES NOT beat the baseline (Macro F1 = 0.3188 vs 0.3359 baseline, 95% CI [0.2578, 0.377] overlaps or fails to improve). Plain conclusion: 2D image classification on scanned ECG plots fails to generalize under a leak-free patient split.