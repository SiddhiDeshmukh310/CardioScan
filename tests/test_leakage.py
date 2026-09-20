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

    def test_every_image_maps_to_exactly_one_split(self):
        split_path = os.path.join('splits', 'split.csv')
        df = pd.read_csv(split_path)
        split_map = dict(zip(df['ecg_id'], df['split']))

        img_dir = os.path.join('data', 'ecg_images')

        assigned_splits = {}
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    m = re.search(r'(\d+)', f)
                    if m:
                        ecg_id = int(m.group(1))
                        if ecg_id in split_map:
                            split = split_map[ecg_id]
                            self.assertIn(split, ['train', 'val', 'test'])
                            if f in assigned_splits:
                                self.assertEqual(assigned_splits[f], split)
                            else:
                                assigned_splits[f] = split

    def test_no_image_ecg_id_in_two_splits(self):
        split_path = os.path.join('splits', 'split.csv')
        df = pd.read_csv(split_path)
        split_map = dict(zip(df['ecg_id'], df['split']))

        img_dir = os.path.join('data', 'ecg_images')

        train_ids, val_ids, test_ids = set(), set(), set()
        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    m = re.search(r'(\d+)', f)
                    if m:
                        ecg_id = int(m.group(1))
                        s = split_map.get(ecg_id)
                        if s == 'train':
                            train_ids.add(ecg_id)
                        elif s == 'val':
                            val_ids.add(ecg_id)
                        elif s == 'test':
                            test_ids.add(ecg_id)

        self.assertEqual(len(train_ids & val_ids), 0, f'Image ecg_id overlap train-val: {train_ids & val_ids}')
        self.assertEqual(len(train_ids & test_ids), 0, f'Image ecg_id overlap train-test: {train_ids & test_ids}')
        self.assertEqual(len(val_ids & test_ids), 0, f'Image ecg_id overlap val-test: {val_ids & test_ids}')

if __name__ == '__main__':
    unittest.main()
