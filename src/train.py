"""
Training script — shared loop for both architectures to keep the comparison fair.

Usage:
    python src/train.py --model resnet18
    python src/train.py --model mobilenet_v2

Both runs use identical splits, augmentation, batch size, optimizer,
learning rate, and epoch budget (all defined in config.py) so that any
accuracy difference reflects the architecture, not the training setup.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
import utils
from dataset import get_dataloaders
from model import get_model, count_trainable_params, count_total_params


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_acc = utils.accuracy_from_logits(logits.detach(), labels)
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(batch_acc, n=images.size(0))

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_meter = utils.AverageMeter()
    acc_meter = utils.AverageMeter()

    for images, labels in tqdm(loader, desc="  val", leave=False):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_acc = utils.accuracy_from_logits(logits, labels)
        loss_meter.update(loss.item(), n=images.size(0))
        acc_meter.update(batch_acc, n=images.size(0))

    return loss_meter.avg, acc_meter.avg


def run_training(model_name: str):
    config.set_seed()
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"Training: {model_name}  |  Device: {device}")
    print(f"{'='*60}")

    # Data
    train_loader, val_loader, _ = get_dataloaders()
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Model
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print(f"Trainable params: {count_trainable_params(model):,} / "
          f"Total params: {count_total_params(model):,}")

    # Loss / optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = config.get_learning_rate(model_name)
    if learning_rate != config.LEARNING_RATE:
        print(f"[NOTE] Using model-specific learning rate for '{model_name}': {learning_rate} "
              f"(default fine-tuning LR is {config.LEARNING_RATE})")
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.WEIGHT_DECAY
    )

    logger = utils.EpochLogger(model_name)
    best_val_acc = 0.0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = utils.timer()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_one_epoch(model, val_loader, criterion, device)

        epoch_time = utils.timer() - epoch_start

        print(f"Epoch {epoch:02d}/{config.NUM_EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"time={epoch_time:.1f}s")

        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, epoch_time)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        utils.save_checkpoint(model, model_name, epoch, val_acc, is_best)

    print(f"\nTraining complete for {model_name}. Best val_acc: {best_val_acc:.4f}")
    utils.plot_loss_curves(model_name)

    # Save a small run summary alongside the checkpoint for the comparison table later
    summary = {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "trainable_params": count_trainable_params(model),
        "total_params": count_total_params(model),
        "epochs": config.NUM_EPOCHS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": learning_rate,
    }
    summary_path = os.path.join(config.REPORTS_DIR, f"{model_name}_train_summary.json")
    utils.save_json(summary, summary_path)
    print(f"Run summary saved to {summary_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 or MobileNetV2 on Tamil digits")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["resnet18", "mobilenet_v2", "simple_cnn"],
        help="Which architecture to train"
    )
    args = parser.parse_args()
    run_training(args.model)


if __name__ == "__main__":
    main()