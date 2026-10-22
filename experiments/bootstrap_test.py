
from datetime import datetime
import json

import numpy as np
import pandas as pd

import argparse
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
)

from config import OUTPUT_DIR
from src.dataset import CLASS_NAMES


MODELS = ("baseline", "weighted_bce")

N_BOOTSTRAP = 2000
RANDOM_SEED = 20261022

METRICS = (
    "macro_auroc",
    "macro_auprc",
    "macro_f1_at_05",
    "macro_f1_validation_thresholds",
)


def safe_mean(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan")

    return float(np.mean(values))


def confidence_interval(values):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return float("nan"), float("nan")

    low, high = np.percentile(values, [2.5, 97.5])

    return float(low), float(high)


def calculate_metrics(y_true, probabilities, thresholds):

    auc_values = []
    ap_values = []
    f1_fixed_values = []
    f1_tuned_values = []

    for j in range(len(CLASS_NAMES)):

        y = y_true[:, j]
        p = probabilities[:, j]

        # AUROC is undefined if a bootstrap sample
        # contains only one label for this class.
        if np.unique(y).size == 2:
            auc_values.append(
                roc_auc_score(y, p)
            )
        else:
            auc_values.append(np.nan)

        # AP is undefined when there are no positive labels.
        if y.sum() > 0:
            ap_values.append(
                average_precision_score(y, p)
            )
        else:
            ap_values.append(np.nan)

        pred_fixed = (p >= 0.5).astype(int)
        pred_tuned = (p >= thresholds[j]).astype(int)

        f1_fixed_values.append(
            f1_score(
                y,
                pred_fixed,
                zero_division=0
            )
        )

        f1_tuned_values.append(
            f1_score(
                y,
                pred_tuned,
                zero_division=0
            )
        )

    return {
        "macro_auroc": safe_mean(auc_values),
        "macro_auprc": safe_mean(ap_values),
        "macro_f1_at_05": float(
            np.mean(f1_fixed_values)
        ),
        "macro_f1_validation_thresholds": float(
            np.mean(f1_tuned_values)
        ),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Run paired patient-level bootstrap analysis."
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="CSV containing per-ECG test predictions."
    )

    parser.add_argument(
        "--threshold-results",
        type=Path,
        required=True,
        help="CSV containing thresholds selected on validation."
    )

    args = parser.parse_args()

    predictions_path = args.predictions
    threshold_results_path = args.threshold_results

    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    if not threshold_results_path.is_file():
        raise FileNotFoundError(threshold_results_path)

    df = pd.read_csv(predictions_path)
    threshold_df = pd.read_csv(threshold_results_path)

    required_columns = {
        "ecg_id",
        "patient_id",
        *[f"true_{c}" for c in CLASS_NAMES],
        *[
            f"{model}_prob_{c}"
            for model in MODELS
            for c in CLASS_NAMES
        ],
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing prediction columns: {sorted(missing)}"
        )

    if df["patient_id"].isna().any():
        raise ValueError("Missing patient IDs.")

    if df["ecg_id"].duplicated().any():
        raise ValueError("Duplicate ECG IDs.")

    # Recover thresholds selected on validation.
    threshold_lookup = {}

    for model in MODELS:

        subset = threshold_df[
            threshold_df["model"] == model
        ]

        if (
            len(subset) != len(CLASS_NAMES)
            or subset["class"].duplicated().any()
            or set(subset["class"]) != set(CLASS_NAMES)
        ):
            raise ValueError(
                f"Invalid validation thresholds for {model}"
            )

        mapping = {
            row["class"]: float(row["best_threshold"])
            for _, row in subset.iterrows()
        }

        threshold_lookup[model] = np.array([
            mapping[c] for c in CLASS_NAMES
        ])

    y_true = df[
        [f"true_{c}" for c in CLASS_NAMES]
    ].to_numpy(dtype=np.int8)

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must be binary.")

    probabilities = {}

    for model in MODELS:

        probabilities[model] = df[
            [f"{model}_prob_{c}" for c in CLASS_NAMES]
        ].to_numpy(dtype=float)

        if not np.isfinite(probabilities[model]).all():
            raise ValueError(
                f"Non-finite predictions for {model}"
            )

        if (
            (probabilities[model] < 0).any()
            or (probabilities[model] > 1).any()
        ):
            raise ValueError(
                f"Invalid probabilities for {model}"
            )

    # Each group contains all ECG rows belonging to one patient.
    patient_groups = [
        group.index.to_numpy(dtype=int)
        for _, group in df.groupby(
            "patient_id",
            sort=False
        )
    ]

    n_patients = len(patient_groups)

    if n_patients < 2:
        raise ValueError("Not enough patients for bootstrap.")

    # Point estimates on the original test set.
    point_estimates = {
        model: calculate_metrics(
            y_true,
            probabilities[model],
            threshold_lookup[model]
        )
        for model in MODELS
    }

    # Collect bootstrap distributions.
    bootstrap_values = {
        f"{model}_{metric}": []
        for model in MODELS
        for metric in METRICS
    }

    bootstrap_values.update({
        f"delta_{metric}": []
        for metric in METRICS
    })

    rng = np.random.default_rng(RANDOM_SEED)

    print("Patient-level paired bootstrap")
    print("Test ECGs:", len(df))
    print("Unique patients:", n_patients)
    print("Bootstrap repetitions:", N_BOOTSTRAP)
    print("Confidence level: 95%")
    print("Random seed:", RANDOM_SEED)

    for iteration in range(N_BOOTSTRAP):

        # Sample patients with replacement.
        sampled_group_indices = rng.integers(
            low=0,
            high=n_patients,
            size=n_patients
        )

        # Keep all ECGs from each selected patient.
        # Repeated patients correctly duplicate their ECG rows.
        bootstrap_indices = np.concatenate([
            patient_groups[i]
            for i in sampled_group_indices
        ])

        y_boot = y_true[bootstrap_indices]

        current = {}

        for model in MODELS:

            p_boot = probabilities[model][bootstrap_indices]

            current[model] = calculate_metrics(
                y_boot,
                p_boot,
                threshold_lookup[model]
            )

            for metric in METRICS:
                bootstrap_values[
                    f"{model}_{metric}"
                ].append(current[model][metric])

        # Paired difference: weighted BCE minus baseline.
        for metric in METRICS:

            delta = (
                current["weighted_bce"][metric]
                - current["baseline"][metric]
            )

            bootstrap_values[
                f"delta_{metric}"
            ].append(delta)

        if (iteration + 1) % 250 == 0:
            print(
                f"Completed {iteration + 1}/{N_BOOTSTRAP}"
            )

    # Summarize estimates and percentile confidence intervals.
    summary_rows = []

    for metric in METRICS:

        baseline_estimate = point_estimates[
            "baseline"
        ][metric]

        weighted_estimate = point_estimates[
            "weighted_bce"
        ][metric]

        delta_estimate = (
            weighted_estimate - baseline_estimate
        )

        baseline_low, baseline_high = confidence_interval(
            bootstrap_values[f"baseline_{metric}"]
        )

        weighted_low, weighted_high = confidence_interval(
            bootstrap_values[f"weighted_bce_{metric}"]
        )

        delta_low, delta_high = confidence_interval(
            bootstrap_values[f"delta_{metric}"]
        )

        summary_rows.append({
            "metric": metric,
            "baseline_estimate": baseline_estimate,
            "baseline_ci95_low": baseline_low,
            "baseline_ci95_high": baseline_high,
            "weighted_estimate": weighted_estimate,
            "weighted_ci95_low": weighted_low,
            "weighted_ci95_high": weighted_high,
            "delta_weighted_minus_baseline": delta_estimate,
            "delta_ci95_low": delta_low,
            "delta_ci95_high": delta_high,
        })

    run_dir = (
        OUTPUT_DIR
        / "runs"
        / (
            "patient_bootstrap_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        run_dir / "bootstrap_summary.csv",
        index=False
    )

    pd.DataFrame(bootstrap_values).to_csv(
        run_dir / "bootstrap_replicates.csv",
        index=False
    )

    metadata = {
        "method": "patient-level paired bootstrap",
        "bootstrap_repetitions": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "confidence_level": 0.95,
        "test_ecgs": int(len(df)),
        "test_patients": int(n_patients),
        "thresholds_source": str(threshold_results_path),
        "predictions_source": str(predictions_path),
        "interpretation": (
            "Differences are weighted_bce minus baseline. "
            "Thresholds were fixed using validation data."
        ),
    }

    with open(
        run_dir / "bootstrap_config.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n===== BOOTSTRAP SUMMARY =====")
    print(summary_df.round(4).to_string(index=False))

    print("\nSaved to:", run_dir)


if __name__ == "__main__":
    main()
