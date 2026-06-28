import os
import sys
import csv
import json
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.models.cifar10_resnet import create_cifar10_resnet18
from src.data.cifar10_loader import get_cifar10_loaders
from src.analysis.mn_cnn_losslandscape import analyze_hessian_spectrum_ave4
from src.visualization.mn_cnn_plots import plot_hessian_spectrum


def load_cifar10_resnet_checkpoint(model_path, device):
    model = create_cifar10_resnet18(pretrained=False)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model


def main():
    analysis_name = "ads_cifar10_resnet18"

    conditions = {
        "checkpoint_20260625_best": os.path.join(
            project_root,
            "results",
            "checkpoints",
            "cifar10_resnet18_20260625-022105",
            "best.pt",
        ),
        "checkpoint_20260627_best": os.path.join(
            project_root,
            "results",
            "checkpoints",
            "cifar10_resnet18_20260627-010605",
            "best.pt",
        ),
    }

    target_layers = [
        "fc",
        "layer1",
    ]

    batch_size = 64
    num_steps = 40
    num_samples = 100
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{run_id}_steps{num_steps}_samples{num_samples}_bs{batch_size}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    out_dir = os.path.join(project_root, "results", "analysis", analysis_name, run_name)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "analysis_name": analysis_name,
        "run_id": run_id,
        "run_name": run_name,
        "conditions": conditions,
        "target_layers": target_layers,
        "batch_size": batch_size,
        "num_steps": num_steps,
        "num_samples": num_samples,
        "device": device,
        "augment": False,
        "use_custom_aug": False,
        "use_random_erasing": False,
    }

    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    train_loader, _ = get_cifar10_loaders(
        BATCH_SIZE=batch_size,
        augment=False,
        use_custom_aug=False,
        use_random_erasing=False,
    )

    criterion = nn.CrossEntropyLoss()
    results = []

    last_eigenvalues = None
    last_l2_sq = None

    for condition_name, model_path in conditions.items():
        if not os.path.exists(model_path):
            print(f"Model file not found, skipped: {model_path}")
            continue

        model = load_cifar10_resnet_checkpoint(model_path, device)
        print(f"Loaded model: {model_path}")

        for target_layer in target_layers:
            print(f"== Condition: {condition_name}, Target layer: {target_layer} ==")

            analysis_result = analyze_hessian_spectrum_ave4(
                model,
                train_loader,
                criterion,
                num_steps=num_steps,
                num_samples=num_samples,
                target_layer=target_layer,
            )

            if analysis_result is None:
                print(f"Hessian analysis failed: {condition_name}, {target_layer}")
                continue

            eigenvalues, max_eigenvector = analysis_result
            max_eig = float(np.max(eigenvalues))

            target_params = [
                (name, p)
                for name, p in model.named_parameters()
                if target_layer in name
            ]

            if not target_params:
                print(f"Target layer not found: {target_layer}")
                continue

            l2_sq = 0.0
            for _, p in target_params:
                l2_sq += float(torch.sum(p.data.cpu().double() ** 2))

            adaptive_sharpness = max_eig * l2_sq

            print(
                f"max_eig={max_eig:.6e}, "
                f"||w||^2={l2_sq:.6e}, "
                f"AdaptiveSharpness={adaptive_sharpness:.6e}"
            )

            current_idx = 0
            print("Eigenvector contribution by parameter:")
            for name, param in target_params:
                num_params = param.numel()
                vec_part = max_eigenvector[current_idx: current_idx + num_params]
                contribution = float(torch.norm(vec_part).item() ** 2)
                print(f"  {name}: contribution={contribution:.6e}")
                current_idx += num_params

            results.append({
                "condition": condition_name,
                "layer": target_layer,
                "max_eigenvalue": max_eig,
                "l2_norm_sq": l2_sq,
                "adaptive_sharpness": adaptive_sharpness,
                "num_steps": num_steps,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "model_path": model_path,
            })

            last_eigenvalues = eigenvalues
            last_l2_sq = l2_sq

            for _, p in model.named_parameters():
                p.requires_grad = True

    out_path = os.path.join(out_dir, "adaptive_sharpness.csv")

    fieldnames = [
        "condition",
        "layer",
        "max_eigenvalue",
        "l2_norm_sq",
        "adaptive_sharpness",
        "num_steps",
        "num_samples",
        "batch_size",
        "model_path",
    ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Saved results to {out_path}")

    if last_eigenvalues is not None and last_l2_sq is not None:
        plot_hessian_spectrum(last_eigenvalues * last_l2_sq)


if __name__ == "__main__":
    main()
