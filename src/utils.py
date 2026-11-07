"""
Shared utilities for training and evaluation.
Keeping these in one place ensures ResNet-18 and MobileNetV2 produce
directly comparable logs, curves, and metrics.
"""

import os
import csv
import json
import time

import torch
import matplotlib.pyplot as plt

import config


# ---------------------------------------------------------------------------
# Epoch-level logging
# ---------------------------------------------------------------------------
class EpochLogger:
    """Accumulates per-epoch metrics and writes them to a CSV as training runs."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.log_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_log.csv")
        self.rows = []

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "epoch_time_sec"]
            )

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_time):
        row = [epoch, train_loss, train_acc, val_loss, val_acc, epoch_time]
        self.rows.append(row)
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def accuracy_from_logits(logits, labels) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


class AverageMeter:
    """Tracks a running average of a metric over a training/eval loop."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, model_name: str, epoch: int, val_acc: float, is_best: bool):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
    }
    last_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_last.pth")
    torch.save(ckpt, last_path)

    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
        torch.save(ckpt, best_path)
        print(f"  -> New best {model_name} checkpoint saved (val_acc={val_acc:.4f})")


def load_checkpoint(model, checkpoint_path: str, device=None):
    device = device or config.DEVICE
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt.get("epoch"), ckpt.get("val_acc")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_loss_curves(model_name: str):
    """Reads the model's training log CSV and plots train/val loss."""
    log_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_log.csv")
    epochs, train_losses, val_losses = [], [], []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="training_loss")
    plt.plot(epochs, val_losses, label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} — Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = os.path.join(config.FIGURES_DIR, f"{model_name}_loss_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def timer():
    return time.time()


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)