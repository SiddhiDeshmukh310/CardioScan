import os
import re
import unittest
import pandas as pd

def extract_ecg_id(filename: str) -> int:
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return int(match.group(1))
    raise ValueError(f'Could not parse ecg_id from filename: {filename}')

class TestLeakage(unittest.TestCase):
    def test_no_ecg_id_overlap_fake_filenames(self):
        fake_train_files = ['1001_3by1.jpg', '1002_6by2.jpg', '1003_12by1.jpg', '1004_3by4.jpg']
        fake_val_files = ['val_2001_3by1.jpg', 'val_2002_6by2.jpg', 'val_2003_3by4.jpg']
        train_ecg_ids = {extract_ecg_id(f) for f in fake_train_files}
        val_ecg_ids = {extract_ecg_id(f) for f in fake_val_files}
        overlap = train_ecg_ids.intersection(val_ecg_ids)
        self.assertEqual(len(overlap), 0, f'Data leakage detected between train and val: {overlap}')

    def test_no_patient_id_overlap_real_ptbxl(self):
        split_path = os.path.join('splits', 'split.csv')
        if not os.path.exists(split_path):
            split_path = os.path.join('D:', 'Antigravity_ECG', 'CardioScan', 'splits', 'split.csv')
        df = pd.read_csv(split_path)
        train_patients = set(df[df['split'] == 'train']['patient_id'].dropna())
        val_patients = set(df[df['split'] == 'val']['patient_id'].dropna())
        test_patients = set(df[df['split'] == 'test']['patient_id'].dropna())
        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)
        self.assertEqual(len(train_val_overlap), 0, f'Patient leakage between train and val: {train_val_overlap}')
        self.assertEqual(len(train_test_overlap), 0, f'Patient leakage between train and test: {train_test_overlap}')
        self.assertEqual(len(val_test_overlap), 0, f'Patient leakage between val and test: {val_test_overlap}')

if __name__ == '__main__':
    unittest.main()
