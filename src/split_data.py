"""
Builds train/val/test CSV splits from data/raw/<class>/*.jpg.

NOTE ON METHODOLOGY:
No writer/source identifiers were recorded during data collection, so this
split is CLASS-STRATIFIED, not writer-independent. This is a documented
limitation: samples from the same writer could in principle appear in more
than one split. If writer metadata becomes available later, re-run splitting
with a group-based method (e.g. sklearn GroupShuffleSplit) instead.

Usage:
    python src/split_data.py
"""

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def collect_image_paths():
    """Walk data/raw/<class>/ and build a DataFrame of (image_path, class)."""
    records = []
    missing_classes = []

    for cls in config.CLASS_NAMES:
        class_dir = os.path.join(config.RAW_DIR, cls)
        if not os.path.isdir(class_dir):
            missing_classes.append(cls)
            continue

        image_paths = (
            glob.glob(os.path.join(class_dir, "*.jpg"))
            + glob.glob(os.path.join(class_dir, "*.jpeg"))
            + glob.glob(os.path.join(class_dir, "*.png"))
        )
        for p in image_paths:
            records.append({"image_path": os.path.abspath(p), "class": cls})

    if missing_classes:
        print(f"[WARNING] No folder found for classes: {missing_classes}. "
              f"Check config.RAW_DIR and config.CLASS_NAMES.")

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(
            f"No images found under {config.RAW_DIR}. "
            f"Expected structure: data/raw/<class>/*.jpg"
        )
    return df


def stratified_split(df: pd.DataFrame):
    """70/15/15 stratified split by class."""
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - config.TRAIN_RATIO),
        stratify=df["class"],
        random_state=config.SPLIT_SEED,
    )

    # temp_df is 30% of data; split it into val (15%) and test (15%) -> 50/50 of temp
    relative_val_ratio = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - relative_val_ratio),
        stratify=temp_df["class"],
        random_state=config.SPLIT_SEED,
    )
    return train_df, val_df, test_df


def report_split_balance(train_df, val_df, test_df):
    print("\nClass balance per split:")
    summary = pd.DataFrame({
        "train": train_df["class"].value_counts(),
        "val": val_df["class"].value_counts(),
        "test": test_df["class"].value_counts(),
    }).fillna(0).astype(int).sort_index()
    print(summary)
    print(f"\nTotal images: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


def main():
    config.set_seed()
    df = collect_image_paths()
    train_df, val_df, test_df = stratified_split(df)

    train_df.to_csv(os.path.join(config.SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(config.SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(config.SPLITS_DIR, "test.csv"), index=False)

    report_split_balance(train_df, val_df, test_df)
    print(f"\nSplits written to: {config.SPLITS_DIR}")
    print("NOTE: stratified (not writer-independent) split — see module docstring.")


if __name__ == "__main__":
    main()