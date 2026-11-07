"""
Evaluation script — run on the held-out test set for a trained checkpoint.

Usage:
    python src/evaluate.py --model resnet18
    python src/evaluate.py --model mobilenet_v2

    # After both models have been evaluated at least once:
    python src/evaluate.py --compare

Produces (per model, under outputs/reports/ and outputs/figures/):
    - {model}_classification_report.csv   (per-class P/R/F1 + macro/weighted avgs)
    - {model}_confusion_matrix.png        (normalized + raw counts)
    - {model}_error_analysis.json         (most-confused class pairs, worst examples)
    - {model}_test_metrics.json           (headline numbers for the comparison table)
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import config
import utils
from dataset import get_dataloaders
from model import get_model, count_trainable_params, count_total_params


@torch.no_grad()
def run_inference(model, loader, device, warmup_batches=3):
    """
    Returns (all_labels, all_preds, all_probs, timing_dict).

    Timing isolates ONLY the model forward pass (model compute), excluding
    data loading and CPU<->GPU transfer, which otherwise dominate the
    measurement and make different-sized models look artificially similar.

    On GPU, CUDA ops are asynchronous -- torch.cuda.synchronize() is required
    before/after timing or Python returns before the GPU has actually
    finished, undercounting the real compute time.
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    forward_times_ms = []
    batch_sizes = []

    is_cuda = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        # Warm-up: first few batches include one-time CUDA kernel compilation
        # and memory allocation overhead. Run them normally but don't time them.
        if batch_idx < warmup_batches:
            logits = model(images)
        else:
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()

            logits = model(images)

            if is_cuda:
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            forward_times_ms.append(elapsed_ms)
            batch_sizes.append(images.size(0))

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.append(labels.numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    total_images_timed = sum(batch_sizes)
    total_forward_ms = sum(forward_times_ms)

    timing = {
        "avg_forward_ms_per_batch": float(np.mean(forward_times_ms)) if forward_times_ms else None,
        "avg_forward_ms_per_image": (total_forward_ms / total_images_timed) if total_images_timed else None,
        "batches_timed": len(forward_times_ms),
        "warmup_batches_excluded": warmup_batches,
        "images_timed": total_images_timed,
        "device": str(device),
    }

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
        timing,
    )


def save_classification_report(model_name, y_true, y_pred):
    report_dict = classification_report(
        y_true, y_pred,
        target_names=config.CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    out_path = os.path.join(config.REPORTS_DIR, f"{model_name}_classification_report.csv")
    report_df.to_csv(out_path)
    print(f"Classification report saved to {out_path}")

    # Print a compact version to console
    print("\nPer-class metrics:")
    print(report_df.loc[config.CLASS_NAMES, ["precision", "recall", "f1-score", "support"]]
          .round(4).to_string())
    print("\nOverall:")
    print(report_df.loc[["accuracy", "macro avg", "weighted avg"]].round(4).to_string())

    return report_df


def save_confusion_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=axes[0]
    )
    axes[0].set_title(f"{model_name} — Confusion Matrix (counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=axes[1]
    )
    axes[1].set_title(f"{model_name} — Confusion Matrix (normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    out_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {out_path}")

    return cm


def error_analysis(model_name, y_true, y_pred, y_probs, top_k=5):
    """Finds the most-confused class pairs and the lowest-confidence correct/incorrect predictions."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # zero out correct predictions to isolate errors

    # Most-confused pairs: (true_class, pred_class, count)
    confused_pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": config.CLASS_NAMES[i],
                    "predicted_class": config.CLASS_NAMES[j],
                    "count": int(cm[i, j]),
                })
    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Lowest-confidence predictions overall (regardless of correctness)
    max_probs = y_probs.max(axis=1)
    lowest_conf_idx = np.argsort(max_probs)[:top_k]
    lowest_confidence_examples = [
        {
            "index": int(idx),
            "true_class": config.CLASS_NAMES[y_true[idx]],
            "predicted_class": config.CLASS_NAMES[y_pred[idx]],
            "confidence": float(max_probs[idx]),
            "correct": bool(y_true[idx] == y_pred[idx]),
        }
        for idx in lowest_conf_idx
    ]

    analysis = {
        "most_confused_pairs": confused_pairs[:top_k],
        "lowest_confidence_examples": lowest_confidence_examples,
        "total_errors": int((y_true != y_pred).sum()),
        "total_samples": int(len(y_true)),
        "error_rate": float((y_true != y_pred).mean()),
    }

    out_path = os.path.join(config.REPORTS_DIR, f"{model_name}_error_analysis.json")
    utils.save_json(analysis, out_path)
    print(f"\nError analysis saved to {out_path}")
    print(f"Total errors: {analysis['total_errors']} / {analysis['total_samples']} "
          f"({analysis['error_rate']*100:.2f}%)")
    if confused_pairs:
        print("Top confused pairs:")
        for pair in confused_pairs[:top_k]:
            print(f"  true={pair['true_class']} -> pred={pair['predicted_class']} "
                  f"({pair['count']}x)")

    return analysis


def run_evaluation(model_name: str, checkpoint_suffix: str = "best"):
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}  |  Device: {device}")
    print(f"{'='*60}")

    # Data — test loader only, unaugmented
    _, _, test_loader = get_dataloaders()

    # Model + checkpoint
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_{checkpoint_suffix}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Run train.py --model {model_name} first."
        )

    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model, epoch, val_acc = utils.load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {epoch} (val_acc={val_acc:.4f})")

    # Inference + timing (for the efficiency comparison)
    # Wall-clock end-to-end time (includes data loading, kept for reference)
    wall_start = time.time()
    y_true, y_pred, y_probs, timing = run_inference(model, test_loader, device)
    wall_time = time.time() - wall_start
    avg_wall_ms = (wall_time / len(y_true)) * 1000

    test_acc = float((y_true == y_pred).mean())
    print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Avg wall-clock time (loader + transfer + compute): {avg_wall_ms:.2f} ms/image")
    print(f"Avg forward-pass time (model compute only, GPU-synced): "
          f"{timing['avg_forward_ms_per_image']:.3f} ms/image "
          f"[{timing['batches_timed']} batches timed, "
          f"{timing['warmup_batches_excluded']} warm-up batches excluded]")

    # Reports
    report_df = save_classification_report(model_name, y_true, y_pred)
    save_confusion_matrix(model_name, y_true, y_pred)
    error_analysis(model_name, y_true, y_pred, y_probs)

    # Headline metrics for the comparison table
    metrics = {
        "model_name": model_name,
        "test_accuracy": test_acc,
        "macro_f1": float(report_df.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report_df.loc["weighted avg", "f1-score"]),
        "macro_precision": float(report_df.loc["macro avg", "precision"]),
        "macro_recall": float(report_df.loc["macro avg", "recall"]),
        "avg_wall_clock_ms_per_image": avg_wall_ms,
        "avg_forward_pass_ms_per_image": timing["avg_forward_ms_per_image"],
        "inference_device": timing["device"],
        "total_params": count_total_params(model),
        "trainable_params": count_trainable_params(model),
        "checkpoint_epoch": epoch,
        "val_acc_at_checkpoint": val_acc,
    }
    metrics_path = os.path.join(config.REPORTS_DIR, f"{model_name}_test_metrics.json")
    utils.save_json(metrics, metrics_path)
    print(f"\nTest metrics saved to {metrics_path}")

    return metrics


def build_comparison_table():
    """Reads {model}_test_metrics.json for every registered model and builds one table."""
    import json

    rows = []
    for model_name in ["resnet18", "mobilenet_v2", "simple_cnn"]:
        path = os.path.join(config.REPORTS_DIR, f"{model_name}_test_metrics.json")
        if not os.path.exists(path):
            print(f"[SKIP] No test metrics found for {model_name} at {path}. "
                  f"Run: python src/evaluate.py --model {model_name}")
            continue
        with open(path, "r") as f:
            rows.append(json.load(f))

    if len(rows) < 2:
        print("\nNeed test metrics for at least two models before comparing. "
              "Run evaluate.py --model <name> for each trained model first.")
        return None

    df = pd.DataFrame(rows).set_index("model_name")
    display_cols = [
        "test_accuracy", "macro_f1", "weighted_f1",
        "avg_forward_pass_ms_per_image", "avg_wall_clock_ms_per_image", "total_params",
    ]
    print("\n" + "=" * 70)
    print("ARCHITECTURE COMPARISON")
    print("=" * 70)
    print(df[display_cols].round(4).to_string())

    out_path = os.path.join(config.REPORTS_DIR, "model_comparison_table.csv")
    df.to_csv(out_path)
    print(f"\nComparison table saved to {out_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set")
    parser.add_argument(
        "--model", type=str, choices=["resnet18", "mobilenet_v2", "simple_cnn"],
        help="Which architecture to evaluate"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Build the side-by-side comparison table from existing test metrics"
    )
    args = parser.parse_args()

    if args.compare:
        build_comparison_table()
    elif args.model:
        run_evaluation(args.model)
    else:
        parser.error("Specify --model {resnet18,mobilenet_v2} or --compare")


if __name__ == "__main__":
    main()