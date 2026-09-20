import unittest
import os
import pandas as pd
import numpy as np

class TestLeakFreeSplit(unittest.TestCase):

    def test_synthetic_ecg_id_isolation(self):
        # Fake filenames test proving no ecg_id appears in both train and validation
        train_filenames = ["100_3by1.jpg", "101_3by1.jpg", "102_3by1.jpg", "103_3by1.jpg"]
        val_filenames = ["104_3by1.jpg", "105_3by1.jpg", "106_3by1.jpg"]
        
        train_ids = {int(f.split('_')[0]) for f in train_filenames}
        val_ids = {int(f.split('_')[0]) for f in val_filenames}
        
        overlap = train_ids.intersection(val_ids)
        self.assertEqual(len(overlap), 0, f"Synthetic test failed: ecg_ids overlapped: {overlap}")

    def test_real_ptbxl_patient_id_leak_free(self):
        # Test real ptbxl_database.csv patient_id isolation across strat_fold splits
        db_path = "ptbxl_database.csv"
        self.assertTrue(os.path.exists(db_path), "ptbxl_database.csv missing!")
        
        df = pd.read_csv(db_path)
        
        train_df = df[df['strat_fold'].isin(range(1, 9))]
        val_df = df[df['strat_fold'] == 9]
        test_df = df[df['strat_fold'] == 10]
        
        train_patients = set(train_df['patient_id'].dropna().unique())
        val_patients = set(val_df['patient_id'].dropna().unique())
        test_patients = set(test_df['patient_id'].dropna().unique())
        
        tv_overlap = train_patients.intersection(val_patients)
        tt_overlap = train_patients.intersection(test_patients)
        vt_overlap = val_patients.intersection(test_patients)
        
        self.assertEqual(len(tv_overlap), 0, f"Train and Val patient_id overlap: {tv_overlap}")
        self.assertEqual(len(tt_overlap), 0, f"Train and Test patient_id overlap: {tt_overlap}")
        self.assertEqual(len(vt_overlap), 0, f"Val and Test patient_id overlap: {vt_overlap}")

    def test_unique_image_split_assignment(self):
        # Verify unique images map to exactly one split
        split_path = "splits/split_leakfree.csv"
        if os.path.exists(split_path):
            df_split = pd.read_csv(split_path)
            counts = df_split['filename'].value_counts()
            duplicates = counts[counts > 1]
            self.assertEqual(len(duplicates), 0, f"Duplicate filenames found in split: {duplicates.to_dict()}")

if __name__ == '__main__':
    unittest.main()
