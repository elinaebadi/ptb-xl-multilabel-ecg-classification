import json

import numpy as np
import pandas as pd
import wfdb

from tqdm import tqdm

from config import (
    DATA_DIR,
    DATASET_DIR,
    METRICS_DIR,
    SAMPLING_RATE,
)


LEAD_NAMES = [
    "I", "II", "III",
    "AVR", "AVL", "AVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6"
]


def main():

    # Load fixed patient-independent splits
    df = pd.read_csv(DATA_DIR / "splits.csv")

    train_df = df[df["split"] == "train"]

    assert len(train_df) == 17084

    # Output file
    output_path = (
        METRICS_DIR / "train_signal_stats.json"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Statistics already exist: {output_path}"
        )

    # Accumulators for each of the 12 leads
    channel_sum = np.zeros(12, dtype=np.float64)
    channel_sum_sq = np.zeros(12, dtype=np.float64)

    total_timesteps = 0

    # Process training ECGs one by one
    for row in tqdm(
        train_df.itertuples(index=False),
        total=len(train_df),
        desc="Computing training statistics"
    ):

        record_path = DATASET_DIR / row.filename_lr

        signal, info = wfdb.rdsamp(
            str(record_path)
        )

        if signal.shape != (1000, 12):
            raise ValueError(
                f"Invalid ECG shape: {record_path}"
            )

        if info["fs"] != SAMPLING_RATE:
            raise ValueError("Invalid sampling rate")

        if info["sig_name"] != LEAD_NAMES:
            raise ValueError("Invalid lead order")

        if not np.isfinite(signal).all():
            raise ValueError(
                f"Non-finite values: {record_path}"
            )

        # Sum across time for each lead
        channel_sum += signal.sum(
            axis=0, dtype=np.float64
        )

        channel_sum_sq += np.square(signal).sum(
            axis=0, dtype=np.float64
        )

        total_timesteps += signal.shape[0]

    # Per-lead mean and standard deviation
    mean = channel_sum / total_timesteps

    variance = (
        channel_sum_sq / total_timesteps
        - mean ** 2
    )

    std = np.sqrt(np.maximum(variance, 0.0))

    if not np.all(std > 0):
        raise ValueError(
            "One or more leads have zero standard deviation."
        )

    # Save statistics
    stats = {
        "source_split": "train",
        "num_records": len(train_df),
        "sampling_rate": SAMPLING_RATE,
        "lead_names": LEAD_NAMES,
        "mean": mean.tolist(),
        "std": std.tolist()
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    # Display results
    results = pd.DataFrame({
        "lead": LEAD_NAMES,
        "mean": mean,
        "std": std
    })

    print("\nTraining Signal Statistics:")
    print(results.round(5).to_string(index=False))

    print("\nRecords processed:", len(train_df))
    print("Total timesteps:", total_timesteps)

    print("\nTraining statistics saved successfully!")
    print("File:", output_path)


if __name__ == "__main__":
    main()
