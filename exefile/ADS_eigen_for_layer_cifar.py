import os
import sys
import csv

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
    conditions = {
        "best": os.path.join(
            project_root,
            "results",
            "checkpoints",
            "cifar10_resnet18_20260625-022105",
            "best.pt",
        ),
        "last": os.path.join(
            project_root,
            "results",
            "checkpoints",
            "cifar10_resnet18_20260625-022105",
            "last.pt",
        ),
    }

    target_layers = [
        "fc",
        # "layer4.1",
        # "layer4",
    ]

    batch_size = 64
    num_steps = 10
    num_samples = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

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

    out_dir = os.path.join(project_root, "results", "csvdata")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "cifar10_resnet18_adaptive_sharpness.csv")

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