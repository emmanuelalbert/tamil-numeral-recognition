"""
Extracts misclassified images from cross-validation out-of-fold predictions,
for manual visual inspection (reviewer request: representative misclassified
samples with an explanation of the visual similarity behind each error).

Copies each misclassified image into outputs/figures/error_samples/<model>/
with a filename encoding the true/predicted labels and confidence, so they're
easy to browse and easy to reference precisely when writing the analysis.

Usage:
    python src/extract_misclassified.py --model resnet18
    python src/extract_misclassified.py --model mobilenet_v2
"""

import argparse
import os
import shutil

import pandas as pd

import config

CLASS_NAMES = config.CLASS_NAMES


def extract(model_name: str, epochs: int = 10):
    csv_path = os.path.join(config.REPORTS_DIR, f"{model_name}_cv_e{epochs}_oof_predictions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run cross_validate.py --model {model_name} "
            f"--epochs {epochs} first."
        )

    df = pd.read_csv(csv_path)
    errors = df[df["true_idx"] != df["pred_idx"]].copy()
    errors["true_class"] = errors["true_idx"].apply(lambda i: CLASS_NAMES[i])
    errors["pred_class"] = errors["pred_idx"].apply(lambda i: CLASS_NAMES[i])
    errors = errors.sort_values("confidence")

    out_dir = os.path.join(config.FIGURES_DIR, "error_samples", model_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{model_name}: {len(errors)} misclassified images out of {len(df)}\n")
    manifest = []
    for _, row in errors.iterrows():
        src = row["image_path"]
        if not os.path.exists(src):
            print(f"  [MISSING] {src}")
            continue
        ext = os.path.splitext(src)[1]
        conf_tag = f"{row['confidence']:.3f}".replace(".", "")
        dst_name = f"true{row['true_class']}_pred{row['pred_class']}_conf{conf_tag}{ext}"
        dst = os.path.join(out_dir, dst_name)
        shutil.copy2(src, dst)
        manifest.append({
            "original_path": src,
            "copied_as": dst_name,
            "true_class": row["true_class"],
            "pred_class": row["pred_class"],
            "confidence": row["confidence"],
            "fold": row["fold"],
        })
        print(f"  true={row['true_class']:>2} pred={row['pred_class']:>2} "
              f"conf={row['confidence']:.3f}  ->  {dst_name}")

    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(out_dir, "_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nCopied {len(manifest)} images to {out_dir}")
    print(f"Manifest saved to {manifest_path}")
    print("\nUpload these images (or the whole error_samples folder) to continue "
          "the error analysis with actual visual inspection.")


def main():
    parser = argparse.ArgumentParser(description="Extract misclassified images for visual review")
    parser.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    extract(args.model, args.epochs)


if __name__ == "__main__":
    main()