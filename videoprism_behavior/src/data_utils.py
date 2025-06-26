import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-extracted VideoPrism features.
    """
    def __init__(self, annotations_file: Path, data_root: Path, split: str = 'all'):
        """
        Args:
            annotations_file: Path to the master annotations CSV. The paths
                              in this file should be relative to `data_root`.
            data_root: The root directory for processed data (e.g., 'videoprism_behavior/processed_data').
            split: The dataset split to use ('train', 'val', 'test', or 'all').
                   This assumes an 'split' column exists in the CSV.
        """
        self.data_root = Path(data_root)
        
        try:
            full_df = pd.read_csv(annotations_file)
        except FileNotFoundError:
            logging.error(f"Annotations file not found at {annotations_file}")
            raise

        # Handle splitting
        if 'split' in full_df.columns and split != 'all':
            self.annotations_df = full_df[full_df['split'] == split].reset_index(drop=True)
        else:
            self.annotations_df = full_df

        if self.annotations_df.empty:
            logging.warning(f"No samples found for split '{split}' in {annotations_file}.")

        # Create mapping from class name to integer ID
        self.classes = sorted(self.annotations_df['label'].unique())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        
        logging.info(f"Loaded {len(self.annotations_df)} samples for split '{split}'.")
        logging.info(f"Found {len(self.classes)} classes: {self.classes}")

    def __len__(self) -> int:
        return len(self.annotations_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Retrieves a feature tensor and its corresponding label.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")
            
        row = self.annotations_df.iloc[idx]
        
        # The path in the CSV is like 'clips/behavior/file.npy'.
        # We replace 'clips' with 'features' to get the correct path.
        feature_relative_path = row['clip_path'].replace('clips', 'features', 1)
        feature_path = self.data_root / feature_relative_path
        
        try:
            # Load feature tensor
            feature_tensor = np.load(feature_path)
        except FileNotFoundError:
            logging.error(f"Feature file not found: {feature_path}")
            # Return a dummy tensor and label to avoid crashing the loader
            # In a real scenario, you might want to handle this more gracefully
            feature_tensor = np.zeros((4096, 768), dtype=np.float32) 
            return torch.from_numpy(feature_tensor).float(), -1

        label_name = row['label']
        label_id = self.class_to_idx[label_name]
        
        return torch.from_numpy(feature_tensor).float(), label_id 