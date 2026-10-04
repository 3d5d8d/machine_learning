import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.analysis.pythia_attention import mean_attention
from src.data.pythia_data import load_records, write_json
from src.models.pythia import FixedPythia, MODEL_ID, REVISION, Pythia


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/pythia_pilot"))
    parser.add_argument("--output", type=Path, default=Path("results/pythia_pilot/step143000"))
    parser.add_argument("--cache", default="models/hf_cache")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--generate-tokens", type=int, default=32)
    args = parser.parse_args()
    records, manifest = load_records(args.data, "train")
    records = records[:args.num_samples]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    normal_model = Pythia.from_pretrained(args.cache, args.device)
    commit = normal_model.config._commit_hash
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=commit, cache_dir=args.cache)
    tokens = torch.tensor(np.array(records[:1]), device=args.device, dtype=torch.long)
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    prompt = tokens[:, :32]
    with torch.no_grad():
        _, normal_loss = normal_model(x, y)
    normal_generation = normal_model.generate(prompt, args.generate_tokens)

    A = mean_attention(normal_model, records, args.device)
    mean_path = args.output / "mean_attention.pt"
    torch.save({"A": A, "model": MODEL_ID, "revision": REVISION, "commit": commit,
                "N": len(records), "length": 2048, "record_ids": manifest["record_ids"][:len(records)],
                "train_sha256": manifest["sha256"], "source": manifest["source"],
                "data_revision": manifest["revision"]}, mean_path)
    fixed_model = FixedPythia(normal_model, A)
    del A
    with torch.no_grad():
        _, fixed_loss = fixed_model(x, y)
    fixed_generation = fixed_model.generate(prompt, args.generate_tokens)

    result = {"model": MODEL_ID, "revision": REVISION, "commit": commit,
              "N": len(records), "mean_attention": str(mean_path),
              "prompt": tokenizer.decode(prompt[0]),
              "normal": {"loss": normal_loss.item(), "generation": tokenizer.decode(normal_generation[0])},
              "fixed": {"loss": fixed_loss.item(), "generation": tokenizer.decode(fixed_generation[0])}}
    write_json(args.output / "result.json", result)
    for name in ("normal", "fixed"):
        print(f"{name}: loss={result[name]['loss']:.6f}")
        print(result[name]["generation"])


if __name__ == "__main__":
    main()
