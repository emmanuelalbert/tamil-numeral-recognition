"""
Builds one unified comparison table across every model that has been evaluated --
CNN models (from evaluate.py) and classical baselines (from classical_baseline.py).

These two families save metrics with slightly different schemas (e.g. CNNs report
avg_forward_pass_ms_per_image, classical methods report avg_predict_ms_per_image
since there's no separate "forward pass" concept for SVM/KNN). This script
normalizes both into one column set rather than requiring identical schemas
upstream, so new baselines can be added later without editing evaluate.py or
classical_baseline.py.

Usage:
    python src/build_comparison_table.py
"""

import glob
import json
import os

import pandas as pd

import config


# Known model categories, used only for a readable "type" column in the output --
# add new model_name values here as new baselines are introduced.
MODEL_TYPE = {
    "resnet18": "Deep CNN (pretrained)",
    "mobilenet_v2": "Deep CNN (pretrained)",
    "vgg11": "Deep CNN (pretrained)",
    "simple_cnn": "Shallow CNN (from scratch)",
    "hog_svm": "Classical ML",
    "hog_knn": "Classical ML",
}

DISPLAY_NAME = {
    "resnet18": "ResNet-18",
    "mobilenet_v2": "MobileNetV2",
    "vgg11": "VGG11",
    "simple_cnn": "Simple CNN",
    "hog_svm": "HOG + SVM",
    "hog_knn": "HOG + KNN",
}


def normalize_record(raw: dict) -> dict:
    """Maps a model's raw test_metrics.json onto one common schema."""
    model_name = raw["model_name"]

    # Per-image inference time: CNNs use "forward pass" (GPU-synced model compute);
    # classical methods use "predict" (no separate GPU/data-loading split, since
    # they run on CPU over pre-extracted features). Normalize both into one column.
    inference_ms = raw.get("avg_forward_pass_ms_per_image")
    if inference_ms is None:
        inference_ms = raw.get("avg_predict_ms_per_image")

    return {
        "model_name": model_name,
        "display_name": DISPLAY_NAME.get(model_name, model_name),
        "model_type": MODEL_TYPE.get(model_name, "Unknown"),
        "test_accuracy": raw.get("test_accuracy"),
        "macro_f1": raw.get("macro_f1"),
        "weighted_f1": raw.get("weighted_f1"),
        "macro_precision": raw.get("macro_precision"),
        "macro_recall": raw.get("macro_recall"),
        "inference_ms_per_image": inference_ms,
        "total_params": raw.get("total_params"),  # None for classical methods -- expected
        "total_errors": raw.get("total_errors"),
        "total_samples": raw.get("total_samples"),
    }


def build_unified_table() -> pd.DataFrame:
    pattern = os.path.join(config.REPORTS_DIR, "*_test_metrics.json")
    metric_files = sorted(glob.glob(pattern))

    if not metric_files:
        raise FileNotFoundError(
            f"No *_test_metrics.json files found in {config.REPORTS_DIR}. "
            f"Run evaluate.py and/or classical_baseline.py first."
        )

    records = []
    for path in metric_files:
        with open(path, "r") as f:
            raw = json.load(f)
        records.append(normalize_record(raw))
        print(f"Loaded: {os.path.basename(path)}")

    df = pd.DataFrame(records).set_index("model_name")

    # Sort by test accuracy, descending -- puts the best performer first,
    # which is generally the most useful default ordering for a paper table.
    df = df.sort_values("test_accuracy", ascending=False)

    return df


def print_table(df: pd.DataFrame):
    display_cols = [
        "display_name", "model_type", "test_accuracy", "macro_f1",
        "weighted_f1", "inference_ms_per_image", "total_params",
        "total_errors", "total_samples",
    ]
    print("\n" + "=" * 100)
    print("UNIFIED BASELINE COMPARISON")
    print("=" * 100)

    printable = df[display_cols].copy()
    printable["test_accuracy"] = (printable["test_accuracy"] * 100).round(2).astype(str) + "%"
    printable["macro_f1"] = printable["macro_f1"].round(4)
    printable["weighted_f1"] = printable["weighted_f1"].round(4)
    printable["inference_ms_per_image"] = printable["inference_ms_per_image"].round(3)
    printable["total_params"] = printable["total_params"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
    )

    print(printable.to_string())


def main():
    df = build_unified_table()
    print_table(df)

    out_path = os.path.join(config.REPORTS_DIR, "unified_comparison_table.csv")
    df.to_csv(out_path)
    print(f"\nUnified comparison table saved to {out_path}")


if __name__ == "__main__":
    main()