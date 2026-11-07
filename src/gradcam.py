"""
Grad-CAM visualization -- addresses the reviewer request for interpretability,
showing which regions of a handwritten digit the network relies on for its
prediction.

Implementation: standard Grad-CAM (Selvaraju et al., 2017) on the last
convolutional block of each architecture:
    - ResNet-18:    model.layer4 (final residual stage, 512 channels, 7x7)
    - MobileNetV2:  model.features[-1] (final 1x1 conv, 1280 channels, 7x7)

For each target image, a forward hook captures that layer's activations and a
backward hook captures the gradient of the predicted class's logit w.r.t. those
activations. Grad-CAM weights each activation channel by the global-average-pooled
gradient, sums, applies ReLU (only positive contributions to the predicted class
are kept), and upsamples to the input resolution to produce a heatmap.

Usage:
    # Grad-CAM for specific images (e.g. the misclassified samples already
    # extracted by extract_misclassified.py):
    python src/gradcam.py --model resnet18 --images path1.jpg path2.jpg

    # Grad-CAM for every image in a manifest CSV (from extract_misclassified.py):
    python src/gradcam.py --model resnet18 --manifest outputs/figures/error_samples/resnet18/_manifest.csv

    # Grad-CAM for N correctly-classified samples per class, for comparison
    # against the misclassified cases:
    python src/gradcam.py --model resnet18 --correct-per-class 2
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import config
import utils
from dataset import get_val_transform
from model import get_model

CLASS_NAMES = config.CLASS_NAMES
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        input_tensor: (1, C, H, W), already on the correct device.
        class_idx: class to explain; if None, uses the model's own top prediction.
        Returns (heatmap as HxW numpy array in [0, 1], predicted_class_idx, confidence).
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred_idx].item())

        target_idx = class_idx if class_idx is not None else pred_idx
        score = logits[0, target_idx]
        score.backward()

        # Global-average-pool the gradients over spatial dims -> per-channel weight
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)  # keep only features that positively support the class

        cam = F.interpolate(
            cam, size=(config.IMAGE_SIZE, config.IMAGE_SIZE),
            mode="bilinear", align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)  # degenerate case: no discriminative signal found

        return cam, pred_idx, confidence


def get_target_layer(model, model_name: str):
    if model_name == "resnet18":
        return model.layer4
    elif model_name == "mobilenet_v2":
        return model.features[-1]
    else:
        raise ValueError(
            f"Grad-CAM target layer not defined for '{model_name}'. "
            f"Add a case in get_target_layer() for this architecture."
        )


# ---------------------------------------------------------------------------
# Overlay + saving
# ---------------------------------------------------------------------------
def make_overlay(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
    """Blends a jet-colormap heatmap over the (resized) original grayscale image."""
    original_resized = original_pil.convert("L").resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE)
    )
    original_rgb = np.stack([np.array(original_resized)] * 3, axis=-1) / 255.0

    heatmap_rgb = cm.jet(heatmap)[:, :, :3]  # drop alpha channel from colormap
    overlay = (1 - alpha) * original_rgb + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    return original_rgb, heatmap_rgb, overlay


def save_gradcam_figure(image_path, original_rgb, heatmap_rgb, overlay,
                         true_class, pred_class, confidence, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    correct = "correct" if true_class == pred_class else "WRONG"
    fig.suptitle(
        f"True: {true_class}  |  Pred: {pred_class} ({confidence:.3f})  [{correct}]",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_model_from_checkpoint(model_name: str, device):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Train this model first.")
    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=False)
    model, epoch, val_acc = utils.load_checkpoint(model, ckpt_path, device)
    model = model.to(device)
    model.eval()
    print(f"Loaded {model_name} checkpoint from epoch {epoch} (val_acc={val_acc:.4f})")
    return model


def run_gradcam_on_images(model_name: str, image_paths, true_classes=None, tag="manual"):
    device = config.DEVICE
    model = load_model_from_checkpoint(model_name, device)
    target_layer = get_target_layer(model, model_name)
    gradcam = GradCAM(model, target_layer)
    transform = get_val_transform()  # no augmentation, matches evaluation preprocessing

    out_dir = os.path.join(config.FIGURES_DIR, "gradcam", model_name, tag)
    os.makedirs(out_dir, exist_ok=True)

    if true_classes is None:
        true_classes = [None] * len(image_paths)

    for path, true_class in zip(image_paths, true_classes):
        if not os.path.exists(path):
            print(f"  [MISSING] {path}")
            continue

        original = Image.open(path).convert("RGB")
        input_tensor = transform(original).unsqueeze(0).to(device)

        heatmap, pred_idx, confidence = gradcam.generate(input_tensor)
        pred_class = CLASS_NAMES[pred_idx]
        true_class_disp = true_class if true_class is not None else "?"

        original_rgb, heatmap_rgb, overlay = make_overlay(original, heatmap)

        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, f"{base}_true{true_class_disp}_pred{pred_class}.png")
        save_gradcam_figure(path, original_rgb, heatmap_rgb, overlay,
                             true_class_disp, pred_class, confidence, out_path)
        print(f"  {os.path.basename(path):40s} true={str(true_class_disp):>2} pred={pred_class:>2} "
              f"conf={confidence:.3f}  ->  {os.path.basename(out_path)}")

    print(f"\nSaved Grad-CAM figures to {out_dir}")


def sample_correct_per_class(model_name: str, n_per_class: int):
    """Picks n correctly-classified test-set images per class for comparison
    against the misclassified Grad-CAMs."""
    device = config.DEVICE
    model = load_model_from_checkpoint(model_name, device)
    transform = get_val_transform()

    test_df = pd.read_csv(os.path.join(config.SPLITS_DIR, "test.csv"))
    chosen_paths, chosen_labels = [], []

    for cls in CLASS_NAMES:
        cls_rows = test_df[test_df["class"].astype(str) == cls]
        found = 0
        for _, row in cls_rows.iterrows():
            if found >= n_per_class:
                break
            img = Image.open(row["image_path"]).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = int(torch.argmax(model(x), dim=1).item())
            if CLASS_NAMES[pred] == cls:
                chosen_paths.append(row["image_path"])
                chosen_labels.append(cls)
                found += 1

    return chosen_paths, chosen_labels


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization")
    parser.add_argument("--model", required=True, choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--images", nargs="+", help="Explicit image paths")
    parser.add_argument("--manifest", type=str,
                         help="CSV from extract_misclassified.py (uses original_path, true_class)")
    parser.add_argument("--correct-per-class", type=int, default=0,
                         help="Also generate Grad-CAM for N correctly-classified images per class")
    args = parser.parse_args()

    if args.images:
        run_gradcam_on_images(args.model, args.images, tag="manual")

    if args.manifest:
        df = pd.read_csv(args.manifest)
        run_gradcam_on_images(
            args.model, df["original_path"].tolist(), df["true_class"].astype(str).tolist(),
            tag="misclassified",
        )

    if args.correct_per_class > 0:
        paths, labels = sample_correct_per_class(args.model, args.correct_per_class)
        run_gradcam_on_images(args.model, paths, labels, tag="correct")

    if not (args.images or args.manifest or args.correct_per_class):
        print("Nothing to do -- specify --images, --manifest, and/or --correct-per-class.")


if __name__ == "__main__":
    main()