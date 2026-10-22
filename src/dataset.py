import json

import numpy as np
import pandas as pd
import torch
import wfdb

from torch.utils.data import Dataset

from config import (
    DATA_DIR,
    DATASET_DIR,
    PROCESSED_DIR,
    TRAIN_STATS_FILE,
)


CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


LEAD_NAMES = [
    "I", "II", "III",
    "AVR", "AVL", "AVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]


class ECGDataset(Dataset):

    def __init__(self, dataframe, normalize=True):

        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.normalize = normalize

        required_columns = [
            "filename_lr",
            *CLASS_NAMES
        ]

        missing = [
            col for col in required_columns
            if col not in self.dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        self.mean = None
        self.std = None

        if self.normalize:

            if not TRAIN_STATS_FILE.is_file():
                raise FileNotFoundError(
                    f"Training statistics not found: "
                    f"{TRAIN_STATS_FILE}"
                )

            with open(
                TRAIN_STATS_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                stats = json.load(f)

            if stats["lead_names"] != LEAD_NAMES:
                raise ValueError(
                    "Lead names do not match training statistics."
                )

            self.mean = np.asarray(
                stats["mean"],
                dtype=np.float32
            ).reshape(1, 12)

            self.std = np.asarray(
                stats["std"],
                dtype=np.float32
            ).reshape(1, 12)

            if (
                self.mean.shape != (1, 12)
                or self.std.shape != (1, 12)
                or not np.isfinite(self.mean).all()
                or not np.isfinite(self.std).all()
                or not np.all(self.std > 0)
            ):
                raise ValueError(
                    "Invalid normalization statistics."
                )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        # Load ECG signal from PTB-XL
        record_path = (
            DATASET_DIR / row["filename_lr"]
        )

        signal, info = wfdb.rdsamp(
            str(record_path)
        )

        # Validate signal structure
        if signal.shape != (1000, 12):
            raise ValueError(
                f"Unexpected signal shape: "
                f"{signal.shape} at {record_path}"
            )

        if info["fs"] != 100:
            raise ValueError(
                f"Unexpected sampling rate: {info['fs']}"
            )

        if info["sig_name"] != LEAD_NAMES:
            raise ValueError(
                f"Unexpected lead order: {info['sig_name']}"
            )

        if not np.isfinite(signal).all():
            raise ValueError(
                f"Non-finite values in ECG: {record_path}"
            )

        # Normalize using training-set statistics
        if self.normalize:
            signal = (signal - self.mean) / self.std

        # Convert (1000, 12) -> (12, 1000)
        signal = signal.T.astype(
            np.float32,
            copy=True
        )

        # Extract five binary labels
        labels = row[CLASS_NAMES].to_numpy(
            dtype=np.float32
        )

        # Convert to PyTorch tensors
        signal_tensor = torch.from_numpy(signal)
        label_tensor = torch.from_numpy(labels)

        return signal_tensor, label_tensor


class CachedECGDataset(Dataset):
    """
    Load normalized ECG signals from a NumPy memory-mapped cache.

    Only train and validation caches are supported.
    The canonical row order comes from data/splits.csv.
    """

    def __init__(self, split):

        if split not in ("train", "val"):
            raise ValueError(
                "Only train and val caches are available."
            )

        self.split = split

        # Load the canonical dataset index
        index_path = DATA_DIR / "splits.csv"

        if not index_path.is_file():
            raise FileNotFoundError(index_path)

        index_df = pd.read_csv(index_path)

        self.dataframe = (
            index_df[index_df["split"] == split]
            .reset_index(drop=True)
            .copy()
        )

        if self.dataframe.empty:
            raise ValueError(
                f"No samples found for split: {split}"
            )

        if self.dataframe["ecg_id"].duplicated().any():
            raise ValueError(
                f"Duplicate ECG IDs in {split} split."
            )

        # Open cache without loading the whole file into RAM
        cache_path = (
            PROCESSED_DIR / f"{split}_signals.npy"
        )

        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)

        self.signals = np.load(
            cache_path,
            mmap_mode="r",
            allow_pickle=False
        )

        expected_shape = (
            len(self.dataframe),
            12,
            1000
        )

        if self.signals.shape != expected_shape:
            raise ValueError(
                f"Expected cache shape {expected_shape}, "
                f"got {self.signals.shape}"
            )

        if self.signals.dtype != np.float32:
            raise ValueError(
                f"Expected float32 cache, got "
                f"{self.signals.dtype}"
            )

        if not all(
            name in self.dataframe.columns
            for name in CLASS_NAMES
        ):
            raise ValueError(
                "Diagnostic labels are missing."
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        # Copy one sample to a writable NumPy array.
        # The full cache remains memory-mapped.
        signal = np.array(
            self.signals[idx],
            dtype=np.float32,
            copy=True
        )

        row = self.dataframe.iloc[idx]

        labels = row[CLASS_NAMES].to_numpy(
            dtype=np.float32,
            copy=True
        )

        return (
            torch.from_numpy(signal),
            torch.from_numpy(labels)
        )
