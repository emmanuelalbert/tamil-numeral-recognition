"""
Central configuration for the Tamil digit recognition project.
Both ResNet-18 and MobileNetV2 pipelines import from here so the
comparison stays fair (same split, same augmentation, same training budget).
"""

import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")              # data/raw/1 ... data/raw/10
SPLITS_DIR = os.path.join(DATA_DIR, "splits")         # train.csv, val.csv, test.csv

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")

for d in [SPLITS_DIR, CHECKPOINT_DIR, LOGS_DIR, FIGURES_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
# Folder names under data/raw/ — adjust if your folders are named differently
# (e.g. "0".."9" vs "1".."10"). This list defines label order everywhere.
CLASS_NAMES = [str(i) for i in range(1, 11)]  # ['1', '2', ..., '10']
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------------
# Split ratios (stratified, class-wise — no writer IDs available)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Image / preprocessing
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
NORMALIZE_MEAN = [0.5]
NORMALIZE_STD = [0.5]

# ---------------------------------------------------------------------------
# Training hyperparameters (shared across both models for a fair comparison)
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4  # for fine-tuned pretrained models (resnet18, mobilenet_v2)
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4  # set to 0 on Windows if you hit multiprocessing issues

# Models trained from scratch (no pretrained weights) need a higher learning
# rate than fine-tuned models -- 1e-4 is tuned for nudging already-good
# pretrained features, not for learning from random initialization in 20
# epochs. This is a standard-practice difference by training regime, not a
# departure from the fairness protocol (batch size, epochs, augmentation,
# splits all stay identical across every model).
LEARNING_RATE_OVERRIDES = {
    "simple_cnn": 1e-3,
}


def get_learning_rate(model_name: str) -> float:
    return LEARNING_RATE_OVERRIDES.get(model_name, LEARNING_RATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False