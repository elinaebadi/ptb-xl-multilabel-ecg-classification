
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import argparse
from pathlib import Path

from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import (
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
)

from src.dataset import CachedECGDataset, CLASS_NAMES
from src.model import ECGCNN


def get_validation_predictions(model, loader, device):

    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.inference_mode():

        for signals, targets in loader:

            signals = signals.to(device)

            logits = model(signals)

            probabilities = torch.sigmoid(logits)

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            all_targets.append(
                targets.numpy()
            )

    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def main():

    parser = argparse.ArgumentParser(
        description="Select per-class thresholds using validation data."
    )

    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        required=True,
        help="Path to the baseline model checkpoint."
    )

    parser.add_argument(
        "--weighted-checkpoint",
        type=Path,
        required=True,
        help="Path to the weighted BCE checkpoint."
    )

    args = parser.parse_args()

    checkpoints = {
        "baseline": args.baseline_checkpoint,
        "weighted_bce": args.weighted_checkpoint,
    }

    device = torch.device("cpu")

    val_dataset = CachedECGDataset("val")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Search thresholds from 0.05 to 0.95 in steps of 0.01.
    thresholds = np.round(
        np.arange(0.05, 0.951, 0.01),
        2
    )

    results = []

    for model_name, checkpoint_path in checkpoints.items():

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        print(f"\nEvaluating: {model_name}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

        if list(checkpoint["class_names"]) != list(CLASS_NAMES):
            raise ValueError(
                "Checkpoint class order does not match."
            )

        model = ECGCNN(
            num_classes=len(CLASS_NAMES)
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        probabilities, targets = get_validation_predictions(
            model,
            val_loader,
            device
        )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Non-finite validation probabilities."
            )

        if not np.isin(targets, [0, 1]).all():
            raise ValueError(
                "Validation targets must be binary."
            )

        best_f1_values = []

        for class_index, class_name in enumerate(CLASS_NAMES):

            y_true = targets[:, class_index]
            y_prob = probabilities[:, class_index]

            f1_at_05 = f1_score(
                y_true,
                y_prob >= 0.5,
                zero_division=0
            )

            candidate_scores = []

            for threshold in thresholds:

                predictions = (
                    y_prob >= threshold
                ).astype(np.int32)

                score = f1_score(
                    y_true,
                    predictions,
                    zero_division=0
                )

                candidate_scores.append(score)

            best_index = int(
                np.argmax(candidate_scores)
            )

            best_threshold = float(
                thresholds[best_index]
            )

            best_f1 = float(
                candidate_scores[best_index]
            )

            best_f1_values.append(best_f1)

            results.append({
                "model": model_name,
                "class": class_name,
                "threshold_at_05": 0.5,
                "f1_at_05": float(f1_at_05),
                "best_threshold": best_threshold,
                "best_validation_f1": best_f1,
                "f1_improvement": (
                    best_f1 - float(f1_at_05)
                ),
            })

        print(
            "Macro-F1 at threshold 0.5:",
            f"{np.mean([r['f1_at_05'] for r in results if r['model'] == model_name]):.4f}"
        )

        print(
            "Macro-F1 with class-specific thresholds:",
            f"{np.mean(best_f1_values):.4f}"
        )

    results_df = pd.DataFrame(results)

    run_dir = (
        OUTPUT_DIR
        / "runs"
        / (
            "threshold_analysis_"
            + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        )
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    output_path = run_dir / "threshold_results.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n===== PER-CLASS THRESHOLD RESULTS =====")

    print(
        results_df.round(4).to_string(index=False)
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
