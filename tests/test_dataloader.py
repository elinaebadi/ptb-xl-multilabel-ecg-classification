
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import src.dataset as dataset_module
from src.dataset import ECGDataset, CLASS_NAMES, LEAD_NAMES


def test_ecg_dataloader_shapes_and_values(tmp_path, monkeypatch):
    # Create synthetic normalization statistics.
    stats_path = tmp_path / "train_signal_stats.json"

    stats = {
        "lead_names": LEAD_NAMES,
        "mean": [0.0] * 12,
        "std": [1.0] * 12,
    }

    stats_path.write_text(
        json.dumps(stats),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        dataset_module, "TRAIN_STATS_FILE", stats_path
    )
    monkeypatch.setattr(
        dataset_module, "DATASET_DIR", tmp_path
    )

    # Mock WFDB so the test does not require real ECG files.
    raw_signal = np.full(
        (1000, 12),
        0.25,
        dtype=np.float32,
    )

    def fake_rdsamp(record_path):
        return raw_signal.copy(), {
            "fs": 100,
            "sig_name": LEAD_NAMES.copy(),
        }

    monkeypatch.setattr(
        dataset_module.wfdb,
        "rdsamp",
        fake_rdsamp,
    )

    # Create 16 synthetic records with binary multi-label targets.
    sample_count = 16

    data = {
        "filename_lr": [
            f"synthetic_record_{i}"
            for i in range(sample_count)
        ],
    }

    for j, class_name in enumerate(CLASS_NAMES):
        data[class_name] = [
            (i + j) % 2
            for i in range(sample_count)
        ]

    dataframe = pd.DataFrame(data)
    dataset = ECGDataset(dataframe, normalize=True)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    signals, labels = next(iter(loader))

    assert signals.shape == (16, 12, 1000)
    assert labels.shape == (16, 5)

    assert signals.dtype == torch.float32
    assert labels.dtype == torch.float32

    assert torch.isfinite(signals).all()
    assert torch.isfinite(labels).all()

    assert torch.all((labels == 0) | (labels == 1))

    # Zero means and unit standard deviations preserve the signal.
    expected_signals = torch.full_like(signals, 0.25)

    torch.testing.assert_close(
        signals,
        expected_signals,
    )

    expected_labels = torch.tensor(
        dataframe[CLASS_NAMES].to_numpy(dtype=np.float32)
    )

    torch.testing.assert_close(labels, expected_labels)
