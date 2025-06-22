import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt, time
from pathlib import Path                    

# ── 0. 画像出力用フォルダー ───────────────────────────
OUTPUT_DIR = Path("results/figs")           
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. MODEL ( 2/3 規模 ) ─────────────────────────────
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)   # 32 → 16
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)  # 64 → 32
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)   # 128 → 64
        self.fc2   = nn.Linear(64, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

# ── 2. DATALOADER (訓練 1/2 に間引き) ───────────────
def mnist_load(batch=64, subset_ratio=0.5):
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train_ds = torchvision.datasets.MNIST('data', True,  download=True, transform=tfm)
    test_ds  = torchvision.datasets.MNIST('data', False, download=True, transform=tfm)

    if subset_ratio < 1.0:                        # ◎ここでデータ量を削減
        sub_len = int(len(train_ds)*subset_ratio)
        train_ds, _ = torch.utils.data.random_split(train_ds, [sub_len, len(train_ds)-sub_len])

    train_dl = torch.utils.data.DataLoader(train_ds, batch, shuffle=True)
    test_dl  = torch.utils.data.DataLoader(test_ds , 512, shuffle=False)
    return train_dl, test_dl

# ── 3. TRAIN / TEST ────────────────────────────────
def run(epoch=3, batch=64, subset=0.5):
    train_dl, test_dl = mnist_load(batch, subset)
    net = TinyNet()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    history = {"train":[], "val":[], "acc":[]}

    for ep in range(1, epoch+1):
        # --- train ---
        net.train(); st=time.time(); loss_sum=0
        for x,y in train_dl:
            opt.zero_grad(); out=net(x); loss=F.nll_loss(out,y); loss.backward(); opt.step()
            loss_sum += loss.item()
        tloss = loss_sum/len(train_dl)

        # --- val ---
        net.eval(); vloss, cor = 0, 0
        with torch.no_grad():
            for x,y in test_dl:
                out = net(x); vloss += F.nll_loss(out,y,reduction='sum').item()
                cor += (out.argmax(1)==y).sum().item()
        vloss /= len(test_dl.dataset); acc = cor/len(test_dl.dataset)
        history["train"].append(tloss); history["val"].append(vloss); history["acc"].append(acc)

        print(f"E{ep}/{epoch}  train {tloss:.3f}  val {vloss:.3f}  acc {acc:.3%}  ({time.time()-st:.1f}s)")

    torch.save(net.state_dict(), "models/mnist_light.pt")
    return history

# ── 4. MAIN & PLOT ────────────────────────────────
if __name__ == "__main__":
    h  = run(epoch=3, batch=64, subset=0.5)
    ep = range(1, len(h["train"]) + 1)

    plt.plot(ep, h["train"], label="train")
    plt.plot(ep, h["val"],   label="val")
    plt.legend(); plt.xlabel("epoch")
    plt.savefig(OUTPUT_DIR / "loss.png")    # ★変更
    plt.close()

    plt.plot(ep, h["acc"])
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.savefig(OUTPUT_DIR / "acc.png")     # ★変更
    plt.close()