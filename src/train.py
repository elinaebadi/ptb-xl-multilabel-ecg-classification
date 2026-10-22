import torch

from src.metrics import calculate_multilabel_metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    max_batches=None
):
    """
    Train the model for one epoch.

    Returns:
        Mean training loss.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_idx, (signals, targets) in enumerate(dataloader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        signals = signals.to(device)
        targets = targets.to(device)

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward pass
        logits = model(signals)

        # Calculate loss
        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss detected.")

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate sample-weighted loss
        batch_size = signals.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty.")

    return total_loss / total_samples


@torch.inference_mode()
def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device,
    threshold=0.5,
    max_batches=None
):
    """
    Evaluate the model without updating weights.

    Metrics are calculated over all collected
    validation samples, not averaged per batch.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_logits = []
    all_targets = []

    for batch_idx, (signals, targets) in enumerate(dataloader):

        if max_batches is not None and batch_idx >= max_batches:
            break

        signals = signals.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(signals)

        loss = criterion(logits, targets)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite validation loss detected.")

        batch_size = signals.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Collect predictions and labels
        all_logits.append(logits.cpu())
        all_targets.append(targets.cpu())

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty.")

    # Concatenate predictions across all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate metrics over the collected dataset
    metrics = calculate_multilabel_metrics(
        logits=all_logits,
        targets=all_targets,
        threshold=threshold
    )

    return {
        "loss": total_loss / total_samples,
        **metrics
    }
