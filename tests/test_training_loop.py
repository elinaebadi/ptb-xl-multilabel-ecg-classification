
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model import ECGCNN
from src.train import train_one_epoch, validate_one_epoch


def make_synthetic_loader(
    num_samples=16,
    batch_size=4,
):
    """Create deterministic synthetic ECG signals and balanced labels."""
    generator = torch.Generator().manual_seed(42)

    signals = torch.randn(
        num_samples,
        12,
        1000,
        generator=generator,
    )

    sample_indices = torch.arange(num_samples)

    targets = torch.stack(
        [
            ((sample_indices + class_idx) % 2).float()
            for class_idx in range(5)
        ],
        dim=1,
    )

    return DataLoader(
        TensorDataset(signals, targets),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def test_training_updates_weights_and_returns_finite_loss():
    torch.manual_seed(42)

    device = torch.device("cpu")
    loader = make_synthetic_loader()

    model = ECGCNN(num_classes=5).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    initial_weights = [
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    train_loss = train_one_epoch(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_batches=2,
    )

    updated_weights = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert torch.isfinite(torch.tensor(train_loss))
    assert train_loss > 0

    assert any(
        not torch.equal(initial, updated)
        for initial, updated in zip(
            initial_weights,
            updated_weights,
        )
    )

    assert model.training


def test_validation_returns_metrics_without_updating_weights():
    torch.manual_seed(42)

    device = torch.device("cpu")
    loader = make_synthetic_loader()

    model = ECGCNN(num_classes=5).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()

    initial_weights = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    results = validate_one_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        threshold=0.5,
    )

    assert torch.isfinite(torch.tensor(results["loss"]))
    assert results["loss"] > 0

    for metric_name in (
        "macro_auroc",
        "macro_auprc",
        "macro_f1",
    ):
        assert metric_name in results
        assert torch.isfinite(
            torch.tensor(results[metric_name])
        )

    assert len(results["per_class"]) == 5

    for initial, current in zip(
        initial_weights,
        model.parameters(),
    ):
        torch.testing.assert_close(
            current,
            initial,
            rtol=0,
            atol=0,
        )

    assert not model.training


def test_training_and_validation_reject_empty_dataloaders():
    device = torch.device("cpu")

    empty_dataset = TensorDataset(
        torch.empty(0, 12, 1000),
        torch.empty(0, 5),
    )

    empty_loader = DataLoader(
        empty_dataset,
        batch_size=4,
    )

    model = ECGCNN(num_classes=5)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    try:
        train_one_epoch(
            model=model,
            dataloader=empty_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Training should reject an empty DataLoader."
        )

    try:
        validate_one_epoch(
            model=model,
            dataloader=empty_loader,
            criterion=criterion,
            device=device,
        )
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Validation should reject an empty DataLoader."
        )
