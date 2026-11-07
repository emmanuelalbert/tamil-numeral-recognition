"""
Dataset and transform pipelines for Tamil digit recognition.

Transform policy (matches manuscript Section 6.2):
  - Train: grayscale -> rotation -> horizontal flip -> resize -> tensor -> normalize
  - Val:   grayscale -> resize -> tensor -> normalize (no flip, no rotation)
  - Test:  grayscale -> resize -> tensor -> normalize only (unbiased evaluation)

Both ResNet-18 and MobileNetV2 expect 3-channel input, so grayscale images
are replicated to 3 channels rather than trained as single-channel.
"""

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class TamilDigitDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.class_to_idx = {c: i for i, c in enumerate(config.CLASS_NAMES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.class_to_idx[str(row["class"])]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN * 3, std=config.NORMALIZE_STD * 3),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.NORMALIZE_MEAN * 3, std=config.NORMALIZE_STD * 3),
    ])


def get_test_transform():
    # Same as val: resize, grayscale, normalize only — no augmentation.
    return get_val_transform()


def get_dataloaders(batch_size: int = None):
    batch_size = batch_size or config.BATCH_SIZE

    train_ds = TamilDigitDataset(
        f"{config.SPLITS_DIR}/train.csv", transform=get_train_transform()
    )
    val_ds = TamilDigitDataset(
        f"{config.SPLITS_DIR}/val.csv", transform=get_val_transform()
    )
    test_ds = TamilDigitDataset(
        f"{config.SPLITS_DIR}/test.csv", transform=get_test_transform()
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader