import numpy as np
import torch

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

from src.dataset import CLASS_NAMES


@torch.no_grad()
def calculate_multilabel_metrics(
    logits,
    targets,
    threshold=0.5
):
    """
    Calculate multi-label classification metrics.

    Args:
        logits: Tensor of shape (N, 5)
        targets: Binary tensor of shape (N, 5)
        threshold: Classification threshold

    Returns:
        Dictionary containing macro and per-class metrics.
    """

    if logits.shape != targets.shape:
        raise ValueError("Logits and targets must have the same shape.")

    if logits.ndim != 2 or logits.shape[1] != len(CLASS_NAMES):
        raise ValueError("Expected shape (N, 5).")

    if logits.shape[0] == 0:
        raise ValueError("Cannot evaluate an empty dataset.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    # Convert logits to probabilities
    probabilities = (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
    )

    y_true = targets.cpu().numpy()

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite prediction probabilities.")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Targets must be binary.")

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    results = {}
    auroc_scores = []
    auprc_scores = []
    f1_scores = []

    # Calculate metrics independently for each class
    for i, class_name in enumerate(CLASS_NAMES):

        true_class = y_true[:, i]
        prob_class = probabilities[:, i]
        pred_class = y_pred[:, i]

        # AUROC requires both positive and negative samples
        if len(np.unique(true_class)) == 2:
            auroc = roc_auc_score(
                true_class, prob_class
            )
        else:
            auroc = float("nan")

        # Average Precision requires positive samples
        if true_class.sum() > 0:
            auprc = average_precision_score(
                true_class, prob_class
            )
        else:
            auprc = float("nan")

        f1 = f1_score(
            true_class,
            pred_class,
            zero_division=0
        )

        results[class_name] = {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "f1": float(f1)
        }

        auroc_scores.append(auroc)
        auprc_scores.append(auprc)
        f1_scores.append(f1)

    # Average over classes with defined scores
    def safe_mean(values):
        valid = np.asarray(values, dtype=float)
        valid = valid[np.isfinite(valid)]

        if len(valid) == 0:
            return float("nan")

        return float(valid.mean())

    return {
        "macro_auroc": safe_mean(auroc_scores),
        "macro_auprc": safe_mean(auprc_scores),
        "macro_f1": safe_mean(f1_scores),
        "per_class": results
    }
