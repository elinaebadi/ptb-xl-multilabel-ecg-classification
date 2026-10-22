
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
from pathlib import Path

from config import OUTPUT_DIR
from src.dataset import CLASS_NAMES


PLOT_DIR = OUTPUT_DIR / "plots"


def main():

    parser = argparse.ArgumentParser(
        description="Plot final ECG classification results."
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        required=True,
        help="Directory containing test_per_class.csv."
    )

    parser.add_argument(
        "--bootstrap-dir",
        type=Path,
        required=True,
        help="Directory containing bootstrap_summary.csv."
    )

    args = parser.parse_args()

    test_dir = args.test_dir
    bootstrap_dir = args.bootstrap_dir

    test_file = test_dir / "test_per_class.csv"
    bootstrap_file = bootstrap_dir / "bootstrap_summary.csv"

    for path in (test_file, bootstrap_file):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required results file not found: {path}"
            )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    per_class = pd.read_csv(test_file)
    bootstrap = pd.read_csv(bootstrap_file)

    baseline = (
        per_class[per_class["model"] == "baseline"]
        .set_index("class")
        .loc[list(CLASS_NAMES)]
    )

    weighted = (
        per_class[per_class["model"] == "weighted_bce"]
        .set_index("class")
        .loc[list(CLASS_NAMES)]
    )

    # Figure 1: Per-class Average Precision
    x = np.arange(len(CLASS_NAMES))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        baseline["average_precision"],
        width,
        label="Baseline"
    )

    ax.bar(
        x + width / 2,
        weighted["average_precision"],
        width,
        label="Weighted BCE"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Precision")
    ax.set_title("Test Performance by Diagnostic Class")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "final_test_per_class_ap.png",
        dpi=200
    )
    plt.close(fig)

    # Figure 2: Per-class F1 at the fixed threshold 0.5
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        baseline["f1_at_05"],
        width,
        label="Baseline"
    )

    ax.bar(
        x + width / 2,
        weighted["f1_at_05"],
        width,
        label="Weighted BCE"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Test F1 by Class (Threshold = 0.5)")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "final_test_per_class_f1.png",
        dpi=200
    )
    plt.close(fig)

    # Figure 3: Paired patient-bootstrap confidence intervals
    labels = {
        "macro_auroc": "Macro-AUROC",
        "macro_auprc": "Macro-AUPRC",
        "macro_f1_at_05": "Macro-F1 (threshold 0.5)",
        "macro_f1_validation_thresholds":
            "Macro-F1 (validation thresholds)",
    }

    selected = bootstrap[
        bootstrap["metric"].isin(labels)
    ].copy()

    selected["label"] = selected["metric"].map(labels)

    selected = selected.set_index("metric").loc[
        list(labels.keys())
    ].reset_index()

    estimates = selected[
        "delta_weighted_minus_baseline"
    ].to_numpy()

    lower = selected["delta_ci95_low"].to_numpy()
    upper = selected["delta_ci95_high"].to_numpy()

    xerr = np.vstack([
        estimates - lower,
        upper - estimates
    ])

    y = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(
        estimates,
        y,
        xerr=xerr,
        fmt="o",
        capsize=4
    )

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(selected["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Weighted BCE − Baseline")
    ax.set_title("Paired Patient-Level Bootstrap: 95% CI")
    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "final_bootstrap_comparison.png",
        dpi=200
    )
    plt.close(fig)

    print("Final figures saved successfully:")

    for name in (
        "final_test_per_class_ap.png",
        "final_test_per_class_f1.png",
        "final_bootstrap_comparison.png",
    ):
        print(PLOT_DIR / name)


if __name__ == "__main__":
    main()
