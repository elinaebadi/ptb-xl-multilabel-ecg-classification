
import json

import numpy as np
import pandas as pd
import torch

import src.dataset as dataset_module
from src.dataset import ECGDataset, CLASS_NAMES, LEAD_NAMES


def test_ecg_normalization_matches_expected_values(
    tmp_path,
    monkeypatch,
):
    # Use different normalization statistics for each lead.
    mean = np.arange(12, dtype=np.float32) * 0.1
    std = np.arange(1, 13, dtype=np.float32)

    stats = {
        "lead_names": LEAD_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }

    stats_path = tmp_path / "train_signal_stats.json"

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

    # Deterministic synthetic ECG with shape (1000, 12).
    raw_signal = np.linspace(
        -1.0,
        1.0,
        12000,
        dtype=np.float32,
    ).reshape(1000, 12)

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

    dataframe = pd.DataFrame({
        "filename_lr": ["synthetic_record"],
        "NORM": [1],
        "MI": [0],
        "STTC": [1],
        "CD": [0],
        "HYP": [0],
    })

    dataset = ECGDataset(
        dataframe,
        normalize=True,
    )

    actual_signal, labels = dataset[0]

    # Independently calculate the expected normalization.
    expected_signal = (
        (raw_signal - mean.reshape(1, 12))
        / std.reshape(1, 12)
    ).T.astype(np.float32)

    np.testing.assert_allclose(
        actual_signal.numpy(),
        expected_signal,
        rtol=1e-5,
        atol=1e-6,
    )

    assert actual_signal.shape == (12, 1000)
    assert labels.shape == (5,)

    assert actual_signal.dtype == torch.float32
    assert labels.dtype == torch.float32

    assert torch.isfinite(actual_signal).all()

    torch.testing.assert_close(
        labels,
        torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32),
    )
