"""
Computational performance analysis -- reports model size, parameter count,
FLOPs/MACs, GPU memory (inference and training step), and inference time.
Training time (per-epoch, total) is pulled from existing outputs/logs/*.csv
rather than re-measured, since re-running training just to time it would be
wasteful and less accurate than the logs already recorded.

Usage:
    python src/compute_profile.py --model resnet18
    python src/compute_profile.py --model mobilenet_v2
    python src/compute_profile.py --model simple_cnn
    python src/compute_profile.py --all
"""

import argparse
import csv
import os

import torch
import torch.nn as nn
from thop import profile as thop_profile

import config
from model import get_model, count_total_params, count_trainable_params


def measure_model_size_mb(model) -> float:
    """Size of the model's state_dict when saved to disk, in MB."""
    tmp_path = os.path.join(config.PROJECT_ROOT, "_tmp_size_check.pth")
    torch.save(model.state_dict(), tmp_path)
    size_mb = os.path.getsize(tmp_path) / (1024 ** 2)
    os.remove(tmp_path)
    return size_mb


def measure_flops_params(model, device):
    """MACs (thop calls them 'flops', they are multiply-accumulates) for one
    224x224x3 image. Reported as GMac and the conventional 2x-GMac FLOPs estimate."""
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    model_copy = model  # thop adds hooks/buffers to the model in place
    macs, _ = thop_profile(model_copy, inputs=(dummy,), verbose=False)
    gmacs = macs / 1e9
    gflops = gmacs * 2  # convention: 1 MAC ~= 2 FLOPs (one mult + one add)
    return gmacs, gflops


def measure_gpu_memory_inference(model, device, batch_size=32):
    if device.type != "cuda":
        return None
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dummy = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mb


def measure_gpu_memory_training_step(model, device, batch_size=32):
    """Peak memory for one forward + backward + optimizer step (worst case,
    since activations for backprop are retained -- this is what determines
    whether a given batch size fits during actual training)."""
    if device.type != "cuda":
        return None
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dummy_x = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    dummy_y = torch.randint(0, config.NUM_CLASSES, (batch_size,)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    optimizer.zero_grad()
    logits = model(dummy_x)
    loss = criterion(logits, dummy_y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return peak_mb


def read_training_time(model_name: str):
    """Sums epoch_time_sec from the model's saved training log, if it exists."""
    log_path = os.path.join(config.LOGS_DIR, f"{model_name}_training_log.csv")
    if not os.path.exists(log_path):
        return None, None
    with open(log_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    total_sec = sum(float(r["epoch_time_sec"]) for r in rows)
    return total_sec, len(rows)


def profile_model(model_name: str):
    device = config.DEVICE
    print(f"\n{'='*60}\nProfiling: {model_name}  |  Device: {device}\n{'='*60}")

    model = get_model(model_name, num_classes=config.NUM_CLASSES, pretrained=True).to(device)

    total_params = count_total_params(model)
    trainable_params = count_trainable_params(model)
    size_mb = measure_model_size_mb(model)

    try:
        gmacs, gflops = measure_flops_params(model, device)
    except Exception as e:
        print(f"[WARNING] FLOPs measurement failed ({e}); reporting None.")
        gmacs, gflops = None, None

    mem_inference = measure_gpu_memory_inference(model, device)
    mem_training = measure_gpu_memory_training_step(model, device)

    train_time_sec, n_epochs_logged = read_training_time(model_name)

    result = {
        "model_name": model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(size_mb, 3),
        "gmacs_per_image": round(gmacs, 4) if gmacs is not None else None,
        "gflops_per_image": round(gflops, 4) if gflops is not None else None,
        "gpu_mem_inference_mb": round(mem_inference, 1) if mem_inference is not None else None,
        "gpu_mem_training_step_mb": round(mem_training, 1) if mem_training is not None else None,
        "total_training_time_sec": round(train_time_sec, 1) if train_time_sec is not None else None,
        "total_training_time_min": round(train_time_sec / 60, 2) if train_time_sec is not None else None,
        "n_epochs_in_log": n_epochs_logged,
        "avg_epoch_time_sec": round(train_time_sec / n_epochs_logged, 2) if train_time_sec else None,
        "device": str(device),
    }

    print(f"Total params:        {total_params:,}")
    print(f"Trainable params:    {trainable_params:,}")
    print(f"Model size on disk:  {size_mb:.2f} MB")
    if gmacs is not None:
        print(f"Computational cost:  {gmacs:.3f} GMac / {gflops:.3f} GFLOPs per {config.IMAGE_SIZE}x{config.IMAGE_SIZE} image")
    if mem_inference is not None:
        print(f"GPU memory (inference, batch={config.BATCH_SIZE}): {mem_inference:.1f} MB")
        print(f"GPU memory (train step, batch={config.BATCH_SIZE}): {mem_training:.1f} MB")
    else:
        print("GPU memory: N/A (no CUDA device)")
    if train_time_sec is not None:
        print(f"Total training time: {train_time_sec/60:.2f} min over {n_epochs_logged} epochs "
              f"({result['avg_epoch_time_sec']:.2f} s/epoch avg)")
    else:
        print(f"Training time: no log found at outputs/logs/{model_name}_training_log.csv")

    out_path = os.path.join(config.REPORTS_DIR, f"{model_name}_compute_profile.json")
    import utils
    utils.save_json(result, out_path)
    print(f"\nSaved to {out_path}")
    return result


def build_summary_table(model_names):
    import json
    import pandas as pd

    rows = []
    for name in model_names:
        path = os.path.join(config.REPORTS_DIR, f"{name}_compute_profile.json")
        if os.path.exists(path):
            with open(path) as f:
                rows.append(json.load(f))
    if not rows:
        print("No compute profiles found to summarize.")
        return
    df = pd.DataFrame(rows).set_index("model_name")
    out_path = os.path.join(config.REPORTS_DIR, "compute_profile_comparison.csv")
    df.to_csv(out_path)
    print(f"\n{'='*70}\nCOMPUTATIONAL PERFORMANCE SUMMARY\n{'='*70}")
    print(df.to_string())
    print(f"\nSaved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Computational performance profiling")
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v2", "simple_cnn"])
    parser.add_argument("--all", action="store_true", help="Profile all three models")
    args = parser.parse_args()

    model_names = ["resnet18", "mobilenet_v2", "simple_cnn"] if args.all else [args.model]
    if not args.model and not args.all:
        parser.error("Specify --model <name> or --all")

    for name in model_names:
        profile_model(name)

    if args.all:
        build_summary_table(model_names)


if __name__ == "__main__":
    main()