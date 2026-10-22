
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import src.dataset as dataset_module
from src.dataset import (
    CLASS_NAMES,
    LEAD_NAMES,
    CachedECGDataset,
    ECGDataset,
)


def prepare_synthetic_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    dataset_dir = data_dir / "raw"

    processed_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)

    mean = np.linspace(-0.2, 0.2, 12, dtype=np.float32)
    std = np.linspace(0.5, 1.5, 12, dtype=np.float32)

    stats_path = data_dir / "train_signal_stats.json"
    stats_path.write_text(
        json.dumps({
            "lead_names": LEAD_NAMES,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }),
        encoding="utf-8",
    )

    rows = []
    raw_signals = {}
    cached_signals = {"train": [], "val": []}

    for split_index, split in enumerate(("train", "val")):
        for i in range(3):
            ecg_id = split_index * 3 + i + 1
            filename = f"record_{ecg_id}"

            rng = np.random.default_rng(ecg_id)
            raw = rng.normal(
                loc=ecg_id * 0.05,
                scale=1.0,
                size=(1000, 12),
            ).astype(np.float32)

            raw_signals[filename] = raw

            normalized = (
                (raw - mean.reshape(1, 12))
                / std.reshape(1, 12)
            ).T.astype(np.float32)

            cached_signals[split].append(normalized)

            row = {
                "ecg_id": ecg_id,
                "split": split,
                "filename_lr": filename,
            }

            row.update({
                name: (ecg_id + j) % 2
                for j, name in enumerate(CLASS_NAMES)
            })

            rows.append(row)

    index_df = pd.DataFrame(rows)
    index_df.to_csv(data_dir / "splits.csv", index=False)

    for split, signals in cached_signals.items():
        np.save(
            processed_dir / f"{split}_signals.npy",
            np.stack(signals).astype(np.float32),
        )

    monkeypatch.setattr(dataset_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        dataset_module, "PROCESSED_DIR", processed_dir
    )
    monkeypatch.setattr(
        dataset_module, "DATASET_DIR", dataset_dir
    )
    monkeypatch.setattr(
        dataset_module, "TRAIN_STATS_FILE", stats_path
    )

    def fake_rdsamp(record_path):
        filename = str(record_path).replace("\\", "/").split("/")[-1]
        return raw_signals[filename].copy(), {
            "fs": 100,
            "sig_name": LEAD_NAMES.copy(),
        }

    monkeypatch.setattr(
        dataset_module.wfdb,
        "rdsamp",
        fake_rdsamp,
    )

    return index_df


def test_cached_signals_match_original_dataset(
    tmp_path,
    monkeypatch,
):
    index_df = prepare_synthetic_data(tmp_path, monkeypatch)

    for split in ("train", "val"):
        split_df = (
            index_df[index_df["split"] == split]
            .reset_index(drop=True)
            .copy()
        )

        original = ECGDataset(split_df, normalize=True)
        cached = CachedECGDataset(split)

        assert (
            original.dataframe["ecg_id"].tolist()
            == cached.dataframe["ecg_id"].tolist()
        )
        assert len(original) == len(cached)

        for idx in range(len(original)):
            original_signal, original_labels = original[idx]
            cached_signal, cached_labels = cached[idx]

            torch.testing.assert_close(
                cached_signal,
                original_signal,
                rtol=1e-5,
                atol=1e-6,
            )

            torch.testing.assert_close(
                cached_labels,
                original_labels,
                rtol=0,
                atol=0,
            )

        signals, labels = next(iter(DataLoader(
            cached,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )))

        assert signals.shape == (2, 12, 1000)
        assert labels.shape == (2, 5)
        assert signals.dtype == torch.float32
        assert labels.dtype == torch.float32
        assert torch.isfinite(signals).all()
        assert torch.isfinite(labels).all()


def test_cached_dataset_rejects_test_split():
    with pytest.raises(ValueError, match="Only train and val"):
        CachedECGDataset("test")


def test_cached_dataset_rejects_wrong_cache_shape(
    tmp_path,
    monkeypatch,
):
    prepare_synthetic_data(tmp_path, monkeypatch)

    wrong_cache = (
        dataset_module.PROCESSED_DIR / "train_signals.npy"
    )

    np.save(
        wrong_cache,
        np.zeros((2, 12, 1000), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Expected cache shape"):
        CachedECGDataset("train")
