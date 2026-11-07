"""
Stratified k-fold cross-validation -- addresses the reviewer request to support the
reported accuracy with repeated experiments (mean +/- std) rather than a single split.

Protocol
  - Pools train/val/test CSVs into one dataset (all images, each used exactly once as test).
  - StratifiedKFold (default 5 folds, shuffled, fixed seed). The fold assignment depends only
    on SPLIT_SEED and the pooled data, so every model sees IDENTICAL folds. This is required
    for paired tests (e.g. McNemar) between models later.
  - Within each fold, ~10% of the non-test portion is held out as validation for checkpoint
    selection (best val accuracy), so the test fold is never used for model selection.
  - Same augmentation, optimizer and batch size as the main experiments. Epoch budget is set
    with --epochs (default: config.NUM_EPOCHS). Learning rate follows config.get_learning_rate.
  - Output folders/files are tagged with the epoch count, so runs with different budgets
    never overwrite each other.

Resumable: each finished fold writes its own metrics/predictions files and is skipped on
re-run (use --force to redo). Fold-level files live in outputs/cv/<model>_k<folds>/.

Usage:
    python src/cross_validate.py --model resnet18 --epochs 10
    python src/cross_validate.py --model mobilenet_v2 --epochs 10
    python src/cross_validate.py --model resnet18 --epochs 10 --summarize-only

NOTE: no writer IDs exist, so folds are class-stratified, not writer-independent. Samples
from one writer can appear in both train and test folds; CV quantifies variance across
splits, not generalization to unseen writers.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader

import config
import utils
from dataset import get_train_transform, get_val_transform
from model import get_model, count_total_params
from train import train_one_epoch, evaluate_one_epoch

VAL_FRACTION = 0.10  # share of the non-test portion used for checkpoint selection
CLASS_TO_IDX = {c: i for i, c in enumerate(config.CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class FrameDataset(Dataset):
    """Dataset built from a DataFrame slice (image_path, class) instead of a CSV file."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["image_path"].tolist()
        self.labels = [CLASS_TO_IDX[str(c)] for c in df["class"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def load_pooled_dataframe() -> pd.DataFrame:
    parts = []
    for split in ("train", "val", "test"):
        path = os.path.join(config.SPLITS_DIR, f"{split}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run split_data.py first.")
        parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True).drop_duplicates(subset="image_path")
    df = df.reset_index(drop=True)
    print(f"Pooled dataset: {len(df)} images")
    print(df["class"].value_counts().sort_index().to_string())
    return df


def make_loader(df, transform, shuffle):
    return DataLoader(
        FrameDataset(df, transform),
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    labels, preds, confs = [], [], []
    for images, y in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        conf, pred = probs.max(dim=1)
        labels.append(y.numpy())
        preds.append(pred.cpu().numpy())
        confs.append(conf.cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(confs)


def run_fold(model_name, fold, df, trainval_idx, test_idx, out_dir, force, epochs):
    metrics_path = os.path.join(out_dir, f"fold_{fold}_metrics.json")
    preds_path = os.path.join(out_dir, f"fold_{fold}_predictions.csv")

    if os.path.exists(metrics_path) and os.path.exists(preds_path) and not force:
        print(f"[Fold {fold}] already completed -- skipping (use --force to rerun)")
        return

    device = config.DEVICE
    trainval_df = df.iloc[trainval_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=VAL_FRACTION,
        stratify=trainval_df["class"],
        random_state=config.SPLIT_SEED + fold,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    print(f"\n{'-'*60}\n[Fold {fold}] train={len(train_df)} val={len(val_df)} test={len(test_df)}"
          f"\n{'-'*60}")

    config.set_seed(config.GLOBAL_SEED + fold)
    train_loader = make_loader(train_df, get_train_transform(), shuffle=True)
    val_loader = make_loader(val_df, get_val_transform(), shuffle=False)
    test_loader = make_loader(test_df, get_val_transform(), shuffle=False)

    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get_learning_rate(model_name),
        weight_decay=config.WEIGHT_DECAY,
    )

    best_val_acc, best_epoch, best_state = -1.0, 0, None
    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate_one_epoch(model, val_loader, criterion, device)
        print(f"  Epoch {epoch:02d}/{epochs} | train_loss={tr_loss:.4f} "
              f"train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_val_acc:  # strict '>' -> earliest best epoch, same rule as train.py
            best_val_acc, best_epoch = va_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
    train_time = time.time() - start

    # Evaluate the best-validation checkpoint on the held-out test fold
    model.load_state_dict(best_state)
    y_true, y_pred, conf = predict(model, test_loader, device)

    report = classification_report(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
        target_names=config.CLASS_NAMES, output_dict=True, zero_division=0,
    )
    metrics = {
        "model_name": model_name,
        "fold": fold,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "test_accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "total_errors": int((y_true != y_pred).sum()),
        "train_time_sec": train_time,
        "epochs": epochs,
        "learning_rate": config.get_learning_rate(model_name),
        "total_params": count_total_params(model),
    }
    utils.save_json(metrics, metrics_path)

    pd.DataFrame({
        "image_path": test_df["image_path"],
        "true_idx": y_true,
        "pred_idx": y_pred,
        "confidence": conf,
        "fold": fold,
    }).to_csv(preds_path, index=False)

    print(f"[Fold {fold}] best epoch={best_epoch} val_acc={best_val_acc:.4f} | "
          f"TEST acc={metrics['test_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} "
          f"errors={metrics['total_errors']}/{len(test_df)} | {train_time/60:.1f} min")

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def summarize(model_name, out_dir, n_folds, epochs):
    prefix = f"{model_name}_cv_e{epochs}"
    rows = []
    for fold in range(1, n_folds + 1):
        path = os.path.join(out_dir, f"fold_{fold}_metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No completed folds found in {out_dir}")
    if len(rows) < n_folds:
        print(f"[WARNING] Only {len(rows)}/{n_folds} folds completed -- summary is partial.")

    fold_df = pd.DataFrame(rows)
    n = len(fold_df)
    summary = {"model_name": model_name, "n_folds_completed": n, "epochs": epochs}

    for col in ["test_accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        vals = fold_df[col].to_numpy()
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if n > 1 else 0.0  # sample std (ddof=1)
        half = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
        summary[col] = {
            "mean": mean, "std": std,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "min": float(vals.min()), "max": float(vals.max()),
        }

    # Pooled out-of-fold predictions: every image predicted exactly once, by a model
    # that never saw it in training.
    preds = pd.concat(
        [pd.read_csv(os.path.join(out_dir, f"fold_{r['fold']}_predictions.csv")) for r in rows],
        ignore_index=True,
    )
    y_true, y_pred = preds["true_idx"].to_numpy(), preds["pred_idx"].to_numpy()
    summary["pooled_oof_accuracy"] = float((y_true == y_pred).mean())
    summary["pooled_oof_errors"] = int((y_true != y_pred).sum())
    summary["pooled_oof_samples"] = int(len(y_true))

    preds.to_csv(os.path.join(config.REPORTS_DIR, f"{prefix}_oof_predictions.csv"), index=False)
    fold_df.to_csv(os.path.join(config.REPORTS_DIR, f"{prefix}_fold_results.csv"), index=False)
    utils.save_json(summary, os.path.join(config.REPORTS_DIR, f"{prefix}_summary.json"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    axes[0].set_title(f"{model_name} — Pooled CV Confusion Matrix (counts)")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES)
    axes[1].set_title(f"{model_name} — Pooled CV Confusion Matrix (normalized)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, f"{prefix}_confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # Console report
    print(f"\n{'='*70}\nCROSS-VALIDATION SUMMARY: {model_name} ({n} folds, {epochs} epochs)\n{'='*70}")
    print(fold_df[["fold", "best_epoch", "test_accuracy", "macro_f1", "total_errors", "n_test"]]
          .round(4).to_string(index=False))
    for col in ["test_accuracy", "macro_f1"]:
        s = summary[col]
        print(f"\n{col}: {s['mean']*100:.2f}% ± {s['std']*100:.2f}%  "
              f"(95% CI [{s['ci95_low']*100:.2f}, {s['ci95_high']*100:.2f}], "
              f"range {s['min']*100:.2f}-{s['max']*100:.2f})")
    print(f"\nPooled out-of-fold: {summary['pooled_oof_accuracy']*100:.2f}% "
          f"({summary['pooled_oof_errors']} errors / {summary['pooled_oof_samples']} images)")
    print(f"\nSaved to {config.REPORTS_DIR} and {config.FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Stratified k-fold cross-validation")
    parser.add_argument("--model", required=True,
                        choices=["resnet18", "mobilenet_v2", "simple_cnn"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help="Epochs per fold (default: config.NUM_EPOCHS)")
    parser.add_argument("--force", action="store_true", help="Rerun folds already completed")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Skip training; summarize completed folds")
    args = parser.parse_args()

    out_dir = os.path.join(config.OUTPUTS_DIR, "cv", f"{args.model}_k{args.folds}_e{args.epochs}")
    os.makedirs(out_dir, exist_ok=True)

    if not args.summarize_only:
        print(f"\n{'='*60}\nCross-validation: {args.model} | {args.folds} folds | "
              f"{args.epochs} epochs | device={config.DEVICE}\n{'='*60}")
        df = load_pooled_dataframe()
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SPLIT_SEED)
        for fold, (trainval_idx, test_idx) in enumerate(skf.split(df, df["class"]), start=1):
            run_fold(args.model, fold, df, trainval_idx, test_idx, out_dir, args.force, args.epochs)

    summarize(args.model, out_dir, args.folds, args.epochs)


if __name__ == "__main__":
    main()