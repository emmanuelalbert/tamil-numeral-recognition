"""
Classical ML baseline — addresses reviewer request for at least one non-deep-learning
comparison method. Uses HOG (Histogram of Oriented Gradients) features + SVM as the
primary classical baseline, with KNN included as a cheap secondary comparison.

Uses the SAME train/val/test splits (data/splits/*.csv) as the CNN models, so the
comparison is on identical samples -- consistent with the fairness constraint used
for the ResNet-18 vs MobileNetV2 comparison.

Usage:
    python src/classical_baseline.py --method svm
    python src/classical_baseline.py --method knn
    python src/classical_baseline.py --method both
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from skimage.transform import resize as sk_resize
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import config
import utils


# ---------------------------------------------------------------------------
# HOG feature extraction
# ---------------------------------------------------------------------------
# Smaller than the 224x224 CNN input -- HOG on classical baselines conventionally
# uses a smaller working resolution since it isn't learning hierarchical features;
# this keeps feature vectors a manageable size and extraction fast on CPU.
HOG_IMAGE_SIZE = (64, 64)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)


def extract_hog_features(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("L")  # grayscale, same as CNN preprocessing
    image = np.array(image)
    image = sk_resize(image, HOG_IMAGE_SIZE, anti_aliasing=True)
    features = hog(image, **HOG_PARAMS)
    return features


def build_feature_matrix(csv_path: str, desc: str = ""):
    df = pd.read_csv(csv_path)
    class_to_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}

    features, labels = [], []
    print(f"Extracting HOG features for {desc} ({len(df)} images)...")
    start = time.time()

    for _, row in df.iterrows():
        feat = extract_hog_features(row["image_path"])
        features.append(feat)
        labels.append(class_to_idx[str(row["class"])])

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({elapsed/len(df)*1000:.1f} ms/image)")

    return np.array(features), np.array(labels)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(method: str):
    method = method.lower()
    if method not in ("svm", "knn"):
        raise ValueError("method must be 'svm' or 'knn'")

    print(f"\n{'='*60}")
    print(f"Classical baseline: HOG + {method.upper()}")
    print(f"{'='*60}")

    # Build feature matrices from the SAME splits the CNNs used
    X_train, y_train = build_feature_matrix(
        os.path.join(config.SPLITS_DIR, "train.csv"), desc="train"
    )
    X_val, y_val = build_feature_matrix(
        os.path.join(config.SPLITS_DIR, "val.csv"), desc="val"
    )
    X_test, y_test = build_feature_matrix(
        os.path.join(config.SPLITS_DIR, "test.csv"), desc="test"
    )

    # Classical methods are sensitive to feature scale -- standardize using
    # train-set statistics only, then apply the same transform to val/test.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # For model selection, combine train+val is common for classical baselines,
    # but to stay consistent with the CNN protocol (train on train, tune/select
    # on val, report on test), we fit on train only and use val as a sanity check.
    if method == "svm":
        # RBF kernel is the standard choice for HOG features on digit-recognition
        # tasks; C and gamma are reasonable defaults, not exhaustively tuned.
        # probability=False (default) since predict_proba is not used here --
        # enabling it adds Platt-scaling overhead to .fit() for no benefit.
        model = SVC(kernel="rbf", C=10, gamma="scale", random_state=config.GLOBAL_SEED)
    else:  # knn
        model = KNeighborsClassifier(n_neighbors=5, weights="distance")

    print(f"\nFitting {method.upper()} on {len(X_train_scaled)} training samples...")
    start = time.time()
    model.fit(X_train_scaled, y_train)
    fit_time = time.time() - start
    print(f"Fit time: {fit_time:.1f}s")

    val_acc = model.score(X_val_scaled, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # Test evaluation, timed the same way as the CNN forward-pass measurement
    # (excluding feature extraction, which is reported separately above)
    start = time.time()
    y_pred = model.predict(X_test_scaled)
    predict_time = time.time() - start
    avg_predict_ms = (predict_time / len(y_test)) * 1000

    test_acc = float((y_pred == y_test).mean())
    print(f"\nTest accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Avg prediction time: {avg_predict_ms:.3f} ms/image (feature extraction excluded)")

    # Classification report -- same format as the CNN evaluate.py output
    model_name = f"hog_{method}"
    report_dict = classification_report(
        y_test, y_pred, target_names=config.CLASS_NAMES, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(config.REPORTS_DIR, f"{model_name}_classification_report.csv")
    report_df.to_csv(report_path)
    print(f"\nClassification report saved to {report_path}")
    print(report_df.loc[config.CLASS_NAMES, ["precision", "recall", "f1-score", "support"]]
          .round(4).to_string())
    print(report_df.loc[["accuracy", "macro avg", "weighted avg"]].round(4).to_string())

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
    )
    plt.title(f"HOG + {method.upper()} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(config.FIGURES_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Metrics JSON -- same schema as the CNN test_metrics.json for easy comparison
    metrics = {
        "model_name": model_name,
        "test_accuracy": test_acc,
        "val_accuracy": float(val_acc),
        "macro_f1": float(report_df.loc["macro avg", "f1-score"]),
        "weighted_f1": float(report_df.loc["weighted avg", "f1-score"]),
        "macro_precision": float(report_df.loc["macro avg", "precision"]),
        "macro_recall": float(report_df.loc["macro avg", "recall"]),
        "avg_predict_ms_per_image": avg_predict_ms,
        "feature_dim": int(X_train.shape[1]),
        "fit_time_sec": fit_time,
        "total_errors": int((y_pred != y_test).sum()),
        "total_samples": int(len(y_test)),
    }
    metrics_path = os.path.join(config.REPORTS_DIR, f"{model_name}_test_metrics.json")
    utils.save_json(metrics, metrics_path)
    print(f"Test metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate classical HOG-based baselines")
    parser.add_argument(
        "--method", type=str, default="svm", choices=["svm", "knn", "both"],
        help="Classical method to run"
    )
    args = parser.parse_args()

    if args.method == "both":
        train_and_evaluate("svm")
        train_and_evaluate("knn")
    else:
        train_and_evaluate(args.method)


if __name__ == "__main__":
    main()