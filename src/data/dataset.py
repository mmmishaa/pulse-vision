import torch.nn as nn
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as v2

from torch.utils.data import Dataset

class RPPGDataset(Dataset):

    def __init__(self, data_path: Path, is_train: bool, df_patients: pd.DataFrame, clip_len: int = 100):

        """
        df_patients columns:
        - patient_id: int
        - step: str ('before' | 'after')
        - frames_file: str ('frames/..._frames.npy')
        - ppg_file: str ('ppg/..._ppg.npy')
        - num_frames: int
        - pulse: float
        - face_detect_ratio: float
        """

        super().__init__()

        self.data_path = data_path
        self.is_train = is_train
        self.samples = []

        for row in df_patients.itertuples():
            self.samples.extend(
                (f"{row.patient_id}_{row.step}_{num_sample}", 
                 row.frames_file,
                 row.ppg_file,
                 num_sample, 
                 num_sample + clip_len,
                 ) for num_sample in range(0, row.num_frames - clip_len + 1, clip_len)
            )

        if self.is_train:
            self.transform_frames = v2.Compose([
                v2.RandomHorizontalFlip(p=0.2),
                v2.ToDtype(torch.float32, scale=True)
            ])
        else:
            self.transform_frames = v2.Compose([
                v2.ToDtype(torch.float32, scale=True)
            ])

    def __len__(self):
        return len(self.samples)

    def _apply_transform(self, frames, ppg):
        ppg = (ppg - ppg.mean()) / (ppg.std() + 1e-7)

        frames = torch.from_numpy(frames)
        ppg = torch.from_numpy(ppg).float()

        frames = frames.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        frames = self.transform_frames(frames)

        return frames, ppg

    def __getitem__(self, idx):
        sample_id, frames_file, ppg_file, start_idx, end_idx = self.samples[idx]

        np_frames = np.load(file=self.data_path / frames_file, mmap_mode="r")[start_idx:end_idx]
        np_ppg = np.load(file=self.data_path / ppg_file, mmap_mode="r")[start_idx:end_idx]

        frames, ppg = self._apply_transform(np_frames, np_ppg)

        return frames, ppg

def data_split(metadata_path: Path, val_ratio: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_patients = pd.read_csv(metadata_path)

    unique_patients = df_patients["patient_id"].unique()
    rng = np.random.default_rng(seed)
    shuffled_patients = rng.permutation(unique_patients)

    val_size = int(len(unique_patients) * val_ratio)
    val_patients = shuffled_patients[:val_size]
    train_patients = shuffled_patients[val_size:]
    
    val_df_patients = df_patients[df_patients["patient_id"].isin(val_patients)].reset_index(drop=True)
    train_df_patients = df_patients[df_patients["patient_id"].isin(train_patients)].reset_index(drop=True)

    return val_df_patients, train_df_patients