from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_landscape_1d(alphas, losses, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(alphas, losses, linewidth=2)
    axis.axvline(0.0, color="red", linestyle="--", alpha=0.7, label="trained model")
    axis.set_title("Transformer Loss Landscape: 1D Slice")
    axis.set_xlabel("alpha")
    axis.set_ylabel("L(theta + alpha d)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_loss_landscape_2d(alphas, betas, loss_grid, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    alpha_grid, beta_grid = np.meshgrid(alphas, betas)

    figure, axis = plt.subplots(figsize=(8, 7))
    contour = axis.contourf(
        alpha_grid,
        beta_grid,
        loss_grid,
        levels=30,
        cmap="viridis",
    )
    axis.scatter([0.0], [0.0], color="red", marker="x", s=80, label="trained model")
    axis.set_title("Transformer Loss Landscape: 2D Slice")
    axis.set_xlabel("alpha")
    axis.set_ylabel("beta")
    axis.legend()
    figure.colorbar(contour, ax=axis, label="Loss")
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_hessian_spectrum(eigenvalues, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 6))
    num_bins = min(30, max(5, len(eigenvalues)))
    axis.hist(eigenvalues, bins=num_bins, density=False)
    axis.axvline(0.0, color="black", linestyle="--", alpha=0.6)
    axis.set_title("Transformer Hessian Ritz-Value Spectrum")
    axis.set_xlabel("Eigenvalue")
    axis.set_ylabel("Count")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
