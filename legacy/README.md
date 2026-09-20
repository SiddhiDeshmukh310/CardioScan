# Legacy Scripts & Archived Experiments

This directory contains legacy scripts and early experimental workflows:
- 	rain_model.py
- 	rain_simple.py
- convert_dataset.py
- ix_dataset.py
- ix_val_and_train.py
- prepare_and_train.py

> [!WARNING]
> **Data Leakage Warning**: These legacy scripts perform random image-level splitting across train and validation sets without grouping by ecg_id or patient_id. Because multiple lead crops from the same recording/patient are assigned to both splits, models trained with these scripts suffer from severe data leakage and produce artificially inflated performance metrics.

For clean, leak-free training grouped by PTB-XL strat_fold, use src/train_efficientnet.py and splits/split.csv.
