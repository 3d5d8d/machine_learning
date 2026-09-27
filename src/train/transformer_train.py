from dataclasses import asdict
from pathlib import Path

import torch

from src.data.tiny_shakespeare import get_shakespeare_loaders
from src.models.nanogpt_ref import GPT, GPTConfig


# ---------- 設定 ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"
CHECKPOINT_PATH = PROJECT_ROOT / "results" / "checkpoints" / "shakespeare_gpt.pt"

BLOCK_SIZE = 128
BATCH_SIZE = 64

MAX_STEPS = 10
EVAL_INTERVAL = 5
EVAL_ITERS = 5

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
SEED = 42


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Validation lossを計算する。"""
    model.eval()
    losses = []

    for batch_index, (x, y) in enumerate(val_loader):
        if batch_index >= EVAL_ITERS:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    # ---------- データ ----------

    text = DATA_PATH.read_text(encoding="utf-8")

    train_loader, val_loader, stoi, itos = get_shakespeare_loaders(
        text=text,
        block_size=BLOCK_SIZE,
        batch_size=BATCH_SIZE,
    )

    print(f"text length: {len(text):,}")
    print(f"vocab size: {len(stoi)}")

    # ---------- モデル ----------

    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=BLOCK_SIZE,
        n_layer=1,
        n_head=4,
        n_embd=128,
        dropout=0.1,
        bias=False,
    )

    model = GPT(config).to(device)

    optimizer = model.configure_optimizers(
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE,
        betas=(0.9, 0.95),
        device_type=device.type,
    )

    # ---------- 学習 ----------

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    train_iterator = iter(train_loader)

    model.train()

    for step in range(1, MAX_STEPS + 1):
        try:
            x, y = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            x, y = next(train_iterator)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        _, loss = model(x, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0:
            val_loss = evaluate(model, val_loader, device)

            print(
                f"step {step:4d} | "
                f"train loss {loss.item():.4f} | "
                f"val loss {val_loss:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_config": asdict(config),
                        "stoi": stoi,
                        "itos": itos,
                        "step": step,
                        "val_loss": val_loss,
                    },
                    CHECKPOINT_PATH,
                )

                print(f"checkpoint saved: {CHECKPOINT_PATH}")

    # ---------- 文章生成 ----------

    model.eval()

    # sorted文字辞書では通常ID 0が改行文字
    context = torch.zeros(
        (1, 1),
        dtype=torch.long,
        device=device,
    )

    generated_ids = model.generate(
        context,
        max_new_tokens=500,
        temperature=0.8,
        top_k=20,
    )

    generated_text = "".join(
        itos[token_id]
        for token_id in generated_ids[0].tolist()
    )

    print("\n--- generated text ---\n")
    print(generated_text)


if __name__ == "__main__":
    main()