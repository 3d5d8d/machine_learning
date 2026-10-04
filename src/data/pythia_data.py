"""Read original training tokens and prepare packed Pile sequences."""
import argparse
import bisect
import hashlib
import io
import json
from pathlib import Path
import struct

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url
from tqdm import tqdm
from transformers import AutoTokenizer

TRAIN_REPO = "EleutherAI/pile-standard-pythia-preshuffled"
MODEL_REPO = "EleutherAI/pythia-70m"
VAL_REPO = "mit-han-lab/pile-val-backup"
VAL_SHA256 = "264c875d8bbd355d8daa9d032b75fd8fb91606218bb84dd1155b203fcd5fab92"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class RemoteTrainingData:
    def __init__(self, revision="main"):
        info = HfApi().dataset_info(TRAIN_REPO, revision=revision, files_metadata=True)
        self.revision = info.sha
        shards = sorted((s.rfilename, s.size) for s in info.siblings if s.rfilename.endswith(".bin"))
        self.names = [name for name, _ in shards]
        self.ends = np.cumsum([size for _, size in shards]).tolist()
        self.session = requests.Session()
        header = self.read("document.idx", 0, 34)
        code = header[17]
        types = {1: "u1", 2: "i1", 3: "<i2", 4: "<i4", 5: "<i8", 8: "<u2"}
        self.dtype = np.dtype(types[code])
        self.count = struct.unpack_from("<Q", header, 18)[0]

    def read(self, name, offset, size):
        url = hf_hub_url(TRAIN_REPO, name, repo_type="dataset", revision=self.revision)
        end = offset + size - 1
        # Include the byte interval in the URL cache key as some CDNs cache redirects.
        with self.session.get(url, params={"range": f"{offset}-{end}"},
                              headers={"Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"},
                              stream=True, timeout=90) as r:
            r.raise_for_status()
            if r.status_code != 206 or not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-{end}/"):
                raise RuntimeError("Server did not honor the exact byte range; refusing a full-shard download")
            result = r.raw.read(size + 1)
        return result

    def record(self, index):
        size = struct.unpack("<i", self.read("document.idx", 34 + index * 4, 4))[0]
        pointer = struct.unpack("<q", self.read("document.idx", 34 + self.count * 4 + index * 8, 8))[0]
        remaining, chunks = size * self.dtype.itemsize, []
        while remaining:
            shard = bisect.bisect_right(self.ends, pointer)
            begin = 0 if shard == 0 else self.ends[shard - 1]
            take = min(remaining, self.ends[shard] - pointer)
            chunks.append(self.read(self.names[shard], pointer - begin, take))
            pointer += take
            remaining -= take
        return np.frombuffer(b"".join(chunks), dtype=self.dtype).astype(np.int64)


def prepare_training(output, count, seed, seen_steps):
    source = RemoteTrainingData()
    stop = min(source.count, seen_steps * 1024)
    indices = np.sort(np.random.default_rng(seed).choice(stop, count, replace=False))
    records = [source.record(int(i)) for i in tqdm(indices, desc="Official training records")]
    path = output / "train.npy"
    np.save(path, np.stack(records))
    write_json(output / "train.json", {
        "source": TRAIN_REPO, "revision": source.revision, "split": "train",
        "record_ids": indices.tolist(), "sampling": "uniform_without_replacement",
        "seed": seed, "seen_steps": seen_steps, "shape": [count, 2049],
        "tokenization": "original_preshuffled_tokens", "sha256": sha256(path),
    })


def prepare_validation(output, count, cache):
    import zstandard
    info = HfApi().dataset_info(VAL_REPO)
    path = hf_hub_download(VAL_REPO, "val.jsonl.zst", repo_type="dataset", revision=info.sha, cache_dir=cache)
    model_sha = HfApi().model_info(MODEL_REPO, revision="step143000").sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=model_sha, cache_dir=cache)
    records, buffer, document_count = [], [], 0
    with open(path, "rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as reader:
        with io.TextIOWrapper(reader, encoding="utf-8") as text:
            for line in text:
                item = json.loads(line)
                document_count += 1
                buffer.extend(tokenizer.encode(item["text"], add_special_tokens=False))
                buffer.append(tokenizer.eos_token_id)
                while len(buffer) >= 2049 and len(records) < count:
                    records.append(buffer[:2049])
                    # Adjacent inputs share only the previous block's final target.
                    buffer = buffer[2048:]
                if len(records) == count:
                    break
    target = output / "eval.npy"
    np.save(target, np.asarray(records, dtype=np.int64))
    write_json(output / "eval.json", {
        "source": VAL_REPO, "revision": info.sha, "split": "validation",
        "upstream": "https://the-eye.eu/public/AI/pile/val.jsonl.zst",
        "source_sha256": VAL_SHA256, "tokenizer_revision": model_sha,
        "sampling": "first_contiguous_packed_blocks_of_validation",
        "documents_read": document_count, "shape": [count, 2049],
        "packing": "text + EOD; stride 2048; no BOS; no document attention reset",
        "sha256": sha256(target),
    })


def load_records(directory, split):
    directory = Path(directory)
    manifest = json.loads((directory / f"{split}.json").read_text(encoding="utf-8"))
    path = directory / f"{split}.npy"
    records = np.load(path, mmap_mode="r", allow_pickle=False)
    return records, manifest


def batches(records, device, length=2048, batch_size=1):
    import torch
    for begin in range(0, len(records), batch_size):
        tokens = torch.tensor(np.array(records[begin:begin + batch_size, :length + 1]), device=device, dtype=torch.long)
        x = tokens[:, :-1].contiguous()
        y = tokens[:, 1:].contiguous()
        yield x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/pythia"))
    parser.add_argument("--cache", default="models/hf_cache")
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--eval-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seen-steps", type=int, default=143000)
    parser.add_argument("--only", choices=["train", "eval", "both"], default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.only in ("train", "both"):
        prepare_training(args.output, args.train_samples, args.seed, args.seen_steps)
    if args.only in ("eval", "both"):
        prepare_validation(args.output, args.eval_samples, args.cache)


if __name__ == "__main__":
    main()
