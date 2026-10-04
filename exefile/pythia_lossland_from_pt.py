import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pythia import MODEL_ID, REVISION, Pythia, FixedPythia
from src.analysis.transformer_losslandscape import compute_1d_loss_landscape, create_random_direction
from src.data.pythia_data import batches, load_records, write_json
from src.models.pythia import MODEL_ID, REVISION, Pythia
from src.visualization.transformer_plots import plot_loss_landscape_1d

DATA_PATH = PROJECT_ROOT / "data" / "pythia_pilot"
CACHE_PATH = PROJECT_ROOT / "models" / "hf_cache"
ANALYSIS_SPLIT = "eval"
BLOCK_SIZE = 2048
BATCH_SIZE = 1
NUM_BATCHES = 2
SEED = 42
LANDSCAPE_RADIUS = 0.1
LANDSCAPE_1D_POINTS = 21
DIRECTION_NORMALIZATION = "parameter"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    #Select no fix or fixed.
    model = Pythia.from_pretrained(cache_dir=CACHE_PATH, device=device)
    #if you want to fixed A, delete comment out
    A_PATH = PROJECT_ROOT / "results/pythia_pilot/step143000/mean_attention.pt"
    saved = torch.load(A_PATH, map_location="cpu", weights_only=True)
    model = FixedPythia(model, saved["A"])
    
    records, manifest = load_records(DATA_PATH, ANALYSIS_SPLIT)
    records = records[:BATCH_SIZE * NUM_BATCHES]
    fixed_batches = list(batches(records, device, BLOCK_SIZE, BATCH_SIZE))
    torch.manual_seed(SEED)
    direction_x = create_random_direction(model, normalization=DIRECTION_NORMALIZATION)

    run_id = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S")
    output_dir = PROJECT_ROOT / "results" / "analysis" / "pythia_losslandscape" / REVISION / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.json", {
        "model": MODEL_ID, "revision": REVISION, "commit": model.config._commit_hash,
        "split": ANALYSIS_SPLIT, "data_sha256": manifest["sha256"],
        "num_sequences": len(records), "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "seed": SEED, "radius": LANDSCAPE_RADIUS, "normalization": DIRECTION_NORMALIZATION,
        "points_1d": LANDSCAPE_1D_POINTS,
        "device": str(device), "dtype": "float32", "torch": str(torch.__version__),
    })
    print(f"checkpoint: {REVISION}, split: {ANALYSIS_SPLIT}, sequences: {len(records)}, device: {device}")

    alphas, losses = compute_1d_loss_landscape(
        model, fixed_batches, direction_x, radius=LANDSCAPE_RADIUS, num_points=LANDSCAPE_1D_POINTS
    )
    np.savetxt(output_dir / "loss_landscape_1d.csv", np.column_stack((alphas, losses)),
               delimiter=",", header="alpha,loss", comments="")
    plot_loss_landscape_1d(alphas, losses, output_dir / "loss_landscape_1d.png")

    print(f"checkpoint loss: {losses[len(losses) // 2]:.6f}")
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
