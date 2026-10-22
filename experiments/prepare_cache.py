
import json
import argparse

import numpy as np
import pandas as pd
import wfdb

from tqdm import tqdm

from config import (
    DATA_DIR,
    DATASET_DIR,
    PROCESSED_DIR,
    TRAIN_STATS_FILE,
    SAMPLING_RATE,
)

from src.dataset import LEAD_NAMES


def main():

    # Load fixed patient-independent splits
    parser = argparse.ArgumentParser(
        description="Prepare normalized ECG signal caches."
    )
    parser.parse_args()

    # Load fixed patient-independent splits
    df = pd.read_csv(DATA_DIR / "splits.csv")

    # Load training-only normalization statistics
    if not TRAIN_STATS_FILE.is_file():
        raise FileNotFoundError(
            f"Missing statistics: {TRAIN_STATS_FILE}"
        )

    with open(
        TRAIN_STATS_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        stats = json.load(f)

    if stats["source_split"] != "train":
        raise ValueError(
            "Normalization statistics must come from train."
        )

    if stats["lead_names"] != LEAD_NAMES:
        raise ValueError("Lead order mismatch.")

    mean = np.asarray(
        stats["mean"], dtype=np.float32
    )

    std = np.asarray(
        stats["std"], dtype=np.float32
    )

    if (
        mean.shape != (12,)
        or std.shape != (12,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not np.all(std > 0)
    ):
        raise ValueError("Invalid normalization statistics.")

    # Only prepare train and validation.
    # The test split remains untouched.
    splits = {
        "train": df[df["split"] == "train"].reset_index(
            drop=True
        ),
        "val": df[df["split"] == "val"].reset_index(
            drop=True
        ),
    }

    expected_counts = {
        "train": 17084,
        "val": 2146,
    }

    for name, split_df in splits.items():
        if len(split_df) != expected_counts[name]:
            raise ValueError(
                f"Unexpected {name} size: {len(split_df)}"
            )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    targets = {
        name: PROCESSED_DIR / f"{name}_signals.npy"
        for name in splits
    }

    # Never silently overwrite an existing cache
    existing = [
        str(path)
        for path in targets.values()
        if path.exists()
    ]

    if existing:
        raise FileExistsError(
            "Cache already exists. Review these files first: "
            + ", ".join(existing)
        )

    temporary = {
        name: PROCESSED_DIR / f".{name}_signals.tmp.npy"
        for name in splits
    }

    # Remove stale temporary files from interrupted runs
    for path in temporary.values():
        path.unlink(missing_ok=True)

    try:
        for split_name, split_df in splits.items():

            temp_path = temporary[split_name]

            signals = np.lib.format.open_memmap(
                temp_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(split_df), 12, 1000)
            )

            try:
                for i, row in enumerate(
                    tqdm(
                        split_df.itertuples(index=False),
                        total=len(split_df),
                        desc=f"Preparing {split_name}"
                    )
                ):

                    record_path = (
                        DATASET_DIR / row.filename_lr
                    )

                    signal, info = wfdb.rdsamp(
                        str(record_path)
                    )

                    if signal.shape != (1000, 12):
                        raise ValueError(
                            f"Invalid shape: {record_path}"
                        )

                    if info["fs"] != SAMPLING_RATE:
                        raise ValueError(
                            f"Invalid sampling rate: {record_path}"
                        )

                    if info["sig_name"] != LEAD_NAMES:
                        raise ValueError(
                            f"Invalid lead order: {record_path}"
                        )

                    if not np.isfinite(signal).all():
                        raise ValueError(
                            f"Non-finite signal: {record_path}"
                        )

                    # Use training-only statistics
                    normalized = (
                        signal - mean
                    ) / std

                    # Store as (12 leads, 1000 timesteps)
                    signals[i] = normalized.T.astype(
                        np.float32,
                        copy=False
                    )

                signals.flush()

            finally:
                del signals

            print(
                f"{split_name}: "
                f"{len(split_df)} signals processed."
            )

        # Publish completed cache files only after both
        # splits have been processed successfully.
        for name in splits:
            temporary[name].replace(targets[name])

    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise

    print("\nCache preparation completed!")

    for name, path in targets.items():
        cached = np.load(
            path,
            mmap_mode="r"
        )

        print(
            f"{name}: shape={cached.shape}, "
            f"dtype={cached.dtype}, "
            f"size={path.stat().st_size / (1024**2):.1f} MiB"
        )

    print("\nThe test split was not processed.")


if __name__ == "__main__":
    main()
