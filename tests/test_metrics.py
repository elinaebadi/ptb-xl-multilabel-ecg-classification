
import torch

from src.metrics import calculate_multilabel_metrics


def test_perfect_predictions_produce_perfect_metrics():
    # Four samples and five diagnostic labels
    targets = torch.tensor(
        [
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.float32,
    )

    # Correct predictions with high confidence
    logits = torch.where(
        targets == 1,
        torch.tensor(8.0),
        torch.tensor(-8.0),
    )

    metrics = calculate_multilabel_metrics(
        logits,
        targets,
        threshold=0.5,
    )

    assert abs(metrics["macro_auroc"] - 1.0) < 1e-6
    assert abs(metrics["macro_auprc"] - 1.0) < 1e-6
    assert abs(metrics["macro_f1"] - 1.0) < 1e-6

    assert len(metrics["per_class"]) == 5

    for class_metrics in metrics["per_class"].values():
        assert abs(class_metrics["auroc"] - 1.0) < 1e-6
        assert abs(class_metrics["auprc"] - 1.0) < 1e-6
        assert abs(class_metrics["f1"] - 1.0) < 1e-6
