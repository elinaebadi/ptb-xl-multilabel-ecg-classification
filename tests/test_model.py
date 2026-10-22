
import torch

from src.model import ECGCNN


def test_ecgcnn_forward_and_backward():
    torch.manual_seed(42)

    model = ECGCNN(num_classes=5)
    x = torch.randn(16, 12, 1000)

    # Forward pass
    logits = model(x)

    assert logits.shape == (16, 5)
    assert torch.isfinite(logits).all()

    # Independent class probabilities
    probabilities = torch.sigmoid(logits)

    assert probabilities.shape == (16, 5)
    assert torch.all(probabilities >= 0)
    assert torch.all(probabilities <= 1)

    # Verify trainable parameters exist
    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    assert trainable_parameters

    # Backward pass
    targets = torch.randint(
        0, 2, (16, 5)
    ).float()

    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(logits, targets)

    assert torch.isfinite(loss)

    loss.backward()

    assert all(
        p.grad is not None
        for p in trainable_parameters
    )

    assert all(
        torch.isfinite(p.grad).all()
        for p in trainable_parameters
    )
