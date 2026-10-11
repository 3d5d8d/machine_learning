import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pythia import Pythia, FixedPythia
from src.data.pythia_data import batches, load_records
from src.analysis.transformer_losslandscape import analyze_hessian_spectrum
from src.visualization.transformer_plots import plot_hessian_spectrum

FIX_ATTENTION = True
NUM_STEPS = 15
TARGET_NAME = None


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Pythia.from_pretrained(
        cache_dir=PROJECT_ROOT / "models/hf_cache", device=device
    )
    if FIX_ATTENTION:
        saved = torch.load(
            PROJECT_ROOT / "results/pythia_pilot/step143000/mean_attention.pt",
            map_location="cpu", weights_only=True
        )
        model = FixedPythia(model, saved["A"])

    records, _ = load_records(PROJECT_ROOT / "data/pythia_pilot", "eval")
    fixed_batches = list(batches(records[:8], device, length=2048, batch_size=1))

    torch.manual_seed(42)
    result = analyze_hessian_spectrum(
        model, fixed_batches, num_steps=NUM_STEPS, target_name=TARGET_NAME
    )

    mode = "fixed" if FIX_ATTENTION else "normal"
    output_dir = PROJECT_ROOT / "results/analysis/pythia_hessian/step143000" / mode
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{mode}: 最大固有値の推定値 = {result.max_eigenvalue:.6e}")
    torch.save({
        "eigenvalues": torch.from_numpy(result.eigenvalues),
        "max_eigenvalue": result.max_eigenvalue,
        "max_eigenvector": result.max_eigenvector,
        "parameter_names": result.parameter_names,
    }, output_dir / "hessian.pt")
    plot_hessian_spectrum(result.eigenvalues, output_dir / "hessian_spectrum.png")


if __name__ == "__main__":
    main()