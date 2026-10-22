
import argparse
import json
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from config import (
    OUTPUT_DIR,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    MIN_LEARNING_RATE,
    VALIDATION_THRESHOLD,
    PRIMARY_METRIC,
    DATA_DIR,
)

from src.dataset import CachedECGDataset, CLASS_NAMES
from src.model import ECGCNN
from src.train import train_one_epoch, validate_one_epoch


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, epoch, metric, settings):

    temporary_path = path.with_suffix(".tmp")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_macro_auprc": metric,
            "class_names": CLASS_NAMES,
            "threshold": VALIDATION_THRESHOLD,
            "config": settings,
        },
        temporary_path
    )

    temporary_path.replace(path)


def main():

    parser = argparse.ArgumentParser(
        description="Train the ECGCNN baseline."
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short integration test without saving a model."
    )

    parser.add_argument(
        "--weighted-bce",
        action="store_true",
        help="Use class-weighted BCE loss."
    )

    args = parser.parse_args()

    seed_everything(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    is_smoke_test = args.smoke_test

    max_epochs = 1 if is_smoke_test else MAX_EPOCHS

    train_batch_limit = 2 if is_smoke_test else None
    val_batch_limit = 5 if is_smoke_test else None

    if args.weighted_bce:
        run_name = (
            "weighted_bce_smoke"
            if is_smoke_test
            else "weighted_bce"
        )
    else:
        run_name = (
            "baseline_smoke"
            if is_smoke_test
            else "baseline"
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = (
        OUTPUT_DIR / "runs" / f"{run_name}_{timestamp}"
    )

    run_dir.mkdir(parents=True, exist_ok=False)

    history_path = run_dir / "history.csv"
    config_path = run_dir / "run_config.json"
    checkpoint_path = run_dir / "baseline_best.pth"

    pos_weight_values = None

    if args.weighted_bce:

        split_df = pd.read_csv(
            DATA_DIR / "splits.csv"
        )

        train_labels = split_df.loc[
            split_df["split"] == "train",
            CLASS_NAMES
        ].to_numpy(dtype=np.float32)

        positive_counts = train_labels.sum(axis=0)
        negative_counts = len(train_labels) - positive_counts

        if np.any(positive_counts <= 0):
            raise ValueError(
                "A class has no positive training samples."
            )

        pos_weight_values = (
            negative_counts / positive_counts
        ).tolist()

    settings = {
        "seed": SEED,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": max_epochs,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "min_delta": MIN_DELTA,
        "primary_metric": PRIMARY_METRIC,
        "threshold": VALIDATION_THRESHOLD,
        "classes": CLASS_NAMES,
        "smoke_test": is_smoke_test,
        "loss": (
            "weighted_bce"
            if args.weighted_bce
            else "bce"
        ),
        "pos_weight": pos_weight_values,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    # Load the verified, normalized caches
    train_dataset = CachedECGDataset("train")
    val_dataset = CachedECGDataset("val")

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        generator=generator,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Model and optimization
    model = ECGCNN(num_classes=len(CLASS_NAMES)).to(device)

    if args.weighted_bce:

        pos_weight_tensor = torch.tensor(
            pos_weight_values,
            dtype=torch.float32,
            device=device
        )

        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor
        )

        print("Loss: Weighted BCE")

        for class_name, weight in zip(
            CLASS_NAMES,
            pos_weight_values
        ):
            print(f"  {class_name}: {weight:.3f}")

    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Loss: Standard BCE")


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        threshold=MIN_DELTA,
        threshold_mode="abs",
        min_lr=MIN_LEARNING_RATE,
    )

    history = []

    best_checkpoint_metric = -float("inf")
    best_stopping_metric = -float("inf")
    best_epoch = None
    epochs_without_improvement = 0

    print("Device:", device)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", max_epochs)
    print("Primary metric:", PRIMARY_METRIC)
    print("Smoke test:", is_smoke_test)
    print("Run directory:", run_dir)

    for epoch in range(1, max_epochs + 1):

        lr_used = optimizer.param_groups[0]["lr"]

        # Training
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_batches=train_batch_limit,
        )

        # Validation: metrics over all collected samples
        val_results = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=VALIDATION_THRESHOLD,
            max_batches=val_batch_limit,
        )

        metric = float(val_results[PRIMARY_METRIC])

        if not np.isfinite(metric):
            raise RuntimeError(
                f"{PRIMARY_METRIC} is not finite."
            )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_macro_auroc": val_results["macro_auroc"],
            "val_macro_auprc": val_results["macro_auprc"],
            "val_macro_f1": val_results["macro_f1"],
            "learning_rate": lr_used,
        }

        for class_name, scores in val_results["per_class"].items():
            for metric_name, score in scores.items():
                row[f"val_{class_name}_{metric_name}"] = score

        history.append(row)

        pd.DataFrame(history).to_csv(
            history_path,
            index=False
        )

        print(
            f"Epoch [{epoch:02d}/{max_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_results['loss']:.4f} | "
            f"Macro AUROC: {val_results['macro_auroc']:.4f} | "
            f"Macro AUPRC: {metric:.4f} | "
            f"Macro F1: {val_results['macro_f1']:.4f} | "
            f"LR: {lr_used:.6g}"
        )

        # Smoke test only checks that the pipeline works.
        # It must not create a misleading model checkpoint.
        if is_smoke_test:
            continue

        # Save the best observed checkpoint.
        if metric > best_checkpoint_metric:
            best_checkpoint_metric = metric
            best_epoch = epoch

            save_checkpoint(
                checkpoint_path,
                model,
                epoch,
                metric,
                settings,
            )

        # Early stopping uses a minimum meaningful improvement.
        if metric > best_stopping_metric + MIN_DELTA:
            best_stopping_metric = metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        scheduler.step(metric)

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "\nEarly stopping triggered. "
                f"Best epoch: {best_epoch}"
            )
            break

    print("\nTraining run completed.")

    print("History:", history_path)
    print("Run configuration:", config_path)

    if is_smoke_test:
        print(
            "Smoke test only: no model checkpoint was saved."
        )
    else:
        print("Best epoch:", best_epoch)
        print(
            "Best Validation Macro-AUPRC:",
            f"{best_checkpoint_metric:.4f}"
        )
        print("Checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
