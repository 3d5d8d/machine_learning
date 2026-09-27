import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.transformer_losslandscape import (
    analyze_hessian_spectrum,
    collect_fixed_batches,
    compute_1d_loss_landscape,
    compute_2d_loss_landscape,
    create_random_direction,
    orthogonalize_direction,
)
from src.data.tiny_shakespeare import get_shakespeare_loaders
from src.models.nanogpt_ref import GPT, GPTConfig
from src.visualization.transformer_plots import (
    plot_hessian_spectrum,
    plot_loss_landscape_1d,
    plot_loss_landscape_2d,
)


CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "shakespeare_gpt.pt"
)
DATA_PATH = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"

ANALYSIS_SPLIT = "train"
ANALYSIS_BATCH_SIZE = 16
NUM_BATCHES = 5
SEED = 42

RUN_1D_LANDSCAPE = True
RUN_2D_LANDSCAPE = False
RUN_HESSIAN = True

LANDSCAPE_RADIUS = 0.5
LANDSCAPE_1D_POINTS = 41
LANDSCAPE_2D_POINTS = 21
DIRECTION_NORMALIZATION = "parameter"

HESSIAN_STEPS = 10
HESSIAN_TARGET_NAME = None


def main():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_PATH}")

    if ANALYSIS_SPLIT not in {"train", "val"}:
        raise ValueError("ANALYSIS_SPLIT must be either 'train' or 'val'")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model_config = GPTConfig(**checkpoint["model_config"])

    model = GPT(model_config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    text = DATA_PATH.read_text(encoding="utf-8")
    train_loader, val_loader, stoi, _ = get_shakespeare_loaders(
        text=text,
        block_size=model_config.block_size,
        batch_size=ANALYSIS_BATCH_SIZE,
    )

    if checkpoint.get("stoi") != stoi:
        raise ValueError("checkpoint vocabulary does not match the current dataset")

    data_loader = train_loader if ANALYSIS_SPLIT == "train" else val_loader
    batches = collect_fixed_batches(
        data_loader,
        device=device,
        num_batches=NUM_BATCHES,
    )

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        PROJECT_ROOT
        / "results"
        / "analysis"
        / "transformer_losslandscape"
        / run_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_config = {
        "run_id": run_id,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_val_loss": checkpoint.get("val_loss"),
        "device": str(device),
        "analysis_split": ANALYSIS_SPLIT,
        "analysis_batch_size": ANALYSIS_BATCH_SIZE,
        "num_batches": NUM_BATCHES,
        "seed": SEED,
        "run_1d_landscape": RUN_1D_LANDSCAPE,
        "run_2d_landscape": RUN_2D_LANDSCAPE,
        "run_hessian": RUN_HESSIAN,
        "landscape_radius": LANDSCAPE_RADIUS,
        "landscape_1d_points": LANDSCAPE_1D_POINTS,
        "landscape_2d_points": LANDSCAPE_2D_POINTS,
        "direction_normalization": DIRECTION_NORMALIZATION,
        "hessian_steps": HESSIAN_STEPS,
        "hessian_target_name": HESSIAN_TARGET_NAME,
        "model_config": checkpoint["model_config"],
    }

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(analysis_config, file, indent=2, ensure_ascii=False)

    direction_x = None

    if RUN_1D_LANDSCAPE or RUN_2D_LANDSCAPE:
        direction_x = create_random_direction(
            model,
            normalization=DIRECTION_NORMALIZATION,
        )

    if RUN_1D_LANDSCAPE:
        alphas, losses = compute_1d_loss_landscape(
            model,
            batches,
            direction_x,
            radius=LANDSCAPE_RADIUS,
            num_points=LANDSCAPE_1D_POINTS,
        )

        _save_1d_csv(output_dir / "loss_landscape_1d.csv", alphas, losses)
        plot_loss_landscape_1d(
            alphas,
            losses,
            output_dir / "loss_landscape_1d.png",
        )
        print(f"1D loss landscape saved: {output_dir}")

    if RUN_2D_LANDSCAPE:
        direction_y = create_random_direction(
            model,
            normalization=DIRECTION_NORMALIZATION,
        )
        direction_y = orthogonalize_direction(direction_y, direction_x)

        alphas, betas, loss_grid = compute_2d_loss_landscape(
            model,
            batches,
            direction_x,
            direction_y,
            radius=LANDSCAPE_RADIUS,
            num_points=LANDSCAPE_2D_POINTS,
        )

        _save_2d_csv(
            output_dir / "loss_landscape_2d.csv",
            alphas,
            betas,
            loss_grid,
        )
        plot_loss_landscape_2d(
            alphas,
            betas,
            loss_grid,
            output_dir / "loss_landscape_2d.png",
        )
        print(f"2D loss landscape saved: {output_dir}")

    if RUN_HESSIAN:
        hessian_result = analyze_hessian_spectrum(
            model,
            batches,
            num_steps=HESSIAN_STEPS,
            target_name=HESSIAN_TARGET_NAME,
        )

        _save_hessian_csv(
            output_dir / "hessian_eigenvalues.csv",
            hessian_result.eigenvalues,
        )
        torch.save(
            {
                "max_eigenvalue": hessian_result.max_eigenvalue,
                "max_eigenvector": hessian_result.max_eigenvector,
                "parameter_names": hessian_result.parameter_names,
                "target_name": HESSIAN_TARGET_NAME,
            },
            output_dir / "hessian_max_eigenpair.pt",
        )
        plot_hessian_spectrum(
            hessian_result.eigenvalues,
            output_dir / "hessian_spectrum.png",
        )
        print(f"maximum Hessian eigenvalue: {hessian_result.max_eigenvalue:.6e}")

    print(f"analysis completed: {output_dir}")


def _save_1d_csv(path, alphas, losses):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "loss"])
        writer.writerows(zip(alphas, losses))


def _save_2d_csv(path, alphas, betas, loss_grid):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["alpha", "beta", "loss"])

        for row, beta in enumerate(betas):
            for column, alpha in enumerate(alphas):
                writer.writerow([alpha, beta, loss_grid[row, column]])


def _save_hessian_csv(path, eigenvalues):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["ritz_value"])
        writer.writerows([[value] for value in eigenvalues])


if __name__ == "__main__":
    main()
