
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import argparse
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from torch.utils.data import DataLoader

from config import (
    DATA_DIR,
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.dataset import ECGDataset, CLASS_NAMES
from src.model import ECGCNN


def safe_mean(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(values.mean())


def main():

    device = torch.device("cpu")

    parser = argparse.ArgumentParser(
        description="Evaluate ECG classifiers on the held-out test set."
    )

    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        required=True,
        help="Path to the baseline checkpoint."
    )

    parser.add_argument(
        "--weighted-checkpoint",
        type=Path,
        required=True,
        help="Path to the weighted BCE checkpoint."
    )

    parser.add_argument(
        "--threshold-results",
        type=Path,
        required=True,
        help="CSV containing thresholds selected on validation."
    )

    args = parser.parse_args()

    checkpoints = {
        "baseline": args.baseline_checkpoint,
        "weighted_bce": args.weighted_checkpoint,
    }

    threshold_results = args.threshold_results

    device = torch.device("cpu")

    # Validate all input paths before evaluation.
    for name, checkpoint_path in checkpoints.items():
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found for {name}: {checkpoint_path}"
            )

    if not threshold_results.is_file():
        raise FileNotFoundError(threshold_results)

    # Load thresholds selected on validation only.
    threshold_df = pd.read_csv(threshold_results)

    required_columns = {
        "model",
        "class",
        "best_threshold",
    }

    if not required_columns.issubset(threshold_df.columns):
        raise ValueError(
            "Threshold results are missing required columns."
        )

    threshold_lookup = {}

    for model_name in checkpoints:

        model_rows = threshold_df[
            threshold_df["model"] == model_name
        ]

        if (
            len(model_rows) != len(CLASS_NAMES)
            or model_rows["class"].duplicated().any()
            or set(model_rows["class"]) != set(CLASS_NAMES)
        ):
            raise ValueError(
                f"Invalid validation thresholds for {model_name}."
            )

        threshold_lookup[model_name] = {
            row["class"]: float(row["best_threshold"])
            for _, row in model_rows.iterrows()
        }

    # Load the untouched test split.
    split_df = pd.read_csv(DATA_DIR / "splits.csv")

    test_df = (
        split_df[split_df["split"] == "test"]
        .reset_index(drop=True)
        .copy()
    )

    if test_df.empty:
        raise ValueError("Test split is empty.")

    if test_df["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs in test split.")

    if "patient_id" in test_df.columns:
        patient_count = test_df["patient_id"].nunique()
    else:
        patient_count = None

    # Use the original WFDB dataset with training-only normalization.
    # No test cache is created.
    test_dataset = ECGDataset(
        test_df,
        normalize=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    models = {}

    # Load both selected checkpoints.
    for model_name, checkpoint_path in checkpoints.items():

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        if list(checkpoint["class_names"]) != list(CLASS_NAMES):
            raise ValueError(
                f"Class order mismatch: {model_name}"
            )

        model = ECGCNN(
            num_classes=len(CLASS_NAMES)
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.eval()

        models[model_name] = {
            "model": model,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "probabilities": [],
        }

    all_targets = []

    print("Final test evaluation")
    print("Device:", device)
    print("Test ECGs:", len(test_dataset))
    print("Test patients:", patient_count)
    print("Models:", ", ".join(checkpoints))
    print("Thresholds: selected on validation")
    print("Do not tune models using these test results.")

    # Read each ECG once and evaluate both models.
    with torch.inference_mode():

        for signals, targets in test_loader:

            signals = signals.to(device)

            all_targets.append(
                targets.numpy()
            )

            for model_name, item in models.items():

                logits = item["model"](signals)

                probabilities = torch.sigmoid(logits)

                item["probabilities"].append(
                    probabilities.cpu().numpy()
                )

    y_true = np.concatenate(all_targets, axis=0)

    per_class_rows = []
    summary_rows = []

    for model_name, item in models.items():

        probabilities = np.concatenate(
            item["probabilities"],
            axis=0
        )

        if probabilities.shape != y_true.shape:
            raise ValueError(
                f"Prediction shape mismatch: {model_name}"
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"Non-finite predictions: {model_name}"
            )

        thresholds = np.asarray([
            threshold_lookup[model_name][class_name]
            for class_name in CLASS_NAMES
        ])

        f1_at_05 = []
        f1_at_tuned = []
        auroc_values = []
        ap_values = []

        for index, class_name in enumerate(CLASS_NAMES):

            y_class = y_true[:, index]
            p_class = probabilities[:, index]

            if len(np.unique(y_class)) < 2:
                auroc = float("nan")
            else:
                auroc = float(
                    roc_auc_score(y_class, p_class)
                )

            if y_class.sum() == 0:
                ap = float("nan")
            else:
                ap = float(
                    average_precision_score(
                        y_class,
                        p_class
                    )
                )

            prediction_05 = (
                p_class >= 0.5
            ).astype(int)

            threshold = float(thresholds[index])

            prediction_tuned = (
                p_class >= threshold
            ).astype(int)

            f1_fixed = float(
                f1_score(
                    y_class,
                    prediction_05,
                    zero_division=0
                )
            )

            f1_tuned = float(
                f1_score(
                    y_class,
                    prediction_tuned,
                    zero_division=0
                )
            )

            precision_tuned = float(
                precision_score(
                    y_class,
                    prediction_tuned,
                    zero_division=0
                )
            )

            recall_tuned = float(
                recall_score(
                    y_class,
                    prediction_tuned,
                    zero_division=0
                )
            )

            auroc_values.append(auroc)
            ap_values.append(ap)
            f1_at_05.append(f1_fixed)
            f1_at_tuned.append(f1_tuned)

            per_class_rows.append({
                "model": model_name,
                "class": class_name,
                "positive_test_samples": int(y_class.sum()),
                "auroc": auroc,
                "average_precision": ap,
                "threshold": threshold,
                "f1_at_05": f1_fixed,
                "f1_at_validation_threshold": f1_tuned,
                "precision_at_validation_threshold": precision_tuned,
                "recall_at_validation_threshold": recall_tuned,
            })

        summary_rows.append({
            "model": model_name,
            "checkpoint_epoch": item["checkpoint_epoch"],
            "test_samples": len(y_true),
            "macro_auroc": safe_mean(auroc_values),
            "macro_auprc": safe_mean(ap_values),
            "macro_f1_at_05": safe_mean(f1_at_05),
            "macro_f1_at_validation_thresholds": safe_mean(
                f1_at_tuned
            ),
        })

    # Save a new, uniquely named evaluation run.
    run_dir = (
        OUTPUT_DIR
        / "runs"
        / (
            "test_evaluation_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    # Save per-ECG predictions for patient-level uncertainty analysis.
    # This does not retrain models or tune any thresholds.

    predictions_df = pd.DataFrame({
        "ecg_id": test_df["ecg_id"].to_numpy(),
        "patient_id": test_df["patient_id"].to_numpy(),
    })

    for i, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"true_{class_name}"] = (
            y_true[:, i].astype(int)
        )

    for model_name, item in models.items():
        probabilities = np.concatenate(
            item["probabilities"],
            axis=0
        )

        for i, class_name in enumerate(CLASS_NAMES):
            predictions_df[
                f"{model_name}_prob_{class_name}"
            ] = probabilities[:, i]

    predictions_df.to_csv(
        run_dir / "test_predictions.csv",
        index=False
    )

    print(
        "Per-ECG predictions saved to:",
        run_dir / "test_predictions.csv"
    )

    per_class_df = pd.DataFrame(per_class_rows)
    summary_df = pd.DataFrame(summary_rows)

    per_class_df.to_csv(
        run_dir / "test_per_class.csv",
        index=False
    )

    summary_df.to_csv(
        run_dir / "test_summary.csv",
        index=False
    )

    report = {
        "split": "test",
        "test_samples": len(test_dataset),
        "test_patients": patient_count,
        "class_names": list(CLASS_NAMES),
        "threshold_source": str(threshold_results),
        "checkpoints": {
            name: str(path)
            for name, path in checkpoints.items()
        },
        "validation_thresholds": threshold_lookup,
    }

    with open(
        run_dir / "evaluation_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    print("\n===== TEST SUMMARY =====")
    print(summary_df.round(4).to_string(index=False))

    print("\n===== TEST PER-CLASS RESULTS =====")
    print(
        per_class_df.round(4).to_string(index=False)
    )

    print("\nResults saved to:", run_dir)


if __name__ == "__main__":
    main()
