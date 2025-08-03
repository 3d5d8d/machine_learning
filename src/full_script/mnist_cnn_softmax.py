import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(0.25)
        self.drop2 = nn.Dropout(0.5)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return F.log_softmax(self.fc2(x), dim=1)

def load_MNIST(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = torchvision.datasets.MNIST("./data", True,  True, transform)
    test_ds  = torchvision.datasets.MNIST("./data", False, True, transform)

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size, shuffle=False)

    return train_dl, test_dl            # ← タプルで返す

def main():
    epochs = 20
    batch_size = 64
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    train_dl, val_dl = load_MNIST(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    net = MyNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    for e in range(epochs):
        # ---- Train ----
        net.train()
        running_loss = 0.0
        for data, target in train_dl:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = net(data)
            loss = F.nll_loss(out, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_dl)
        history["train_loss"].append(train_loss)

        # ---- Validation ----
        net.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device), target.to(device)
                out = net(data)
                val_loss += F.nll_loss(out, target, reduction='sum').item()
                pred = out.argmax(1)
                correct += (pred == target).sum().item()
        val_loss /= len(val_dl.dataset)
        acc = correct / len(val_dl.dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)

        print(f"Epoch {e+1:02}/{epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | acc {acc:.3%}")

    torch.save(net.state_dict(), "my_mnist_model.pt")
    # ---- 省略：プロット部はそのまま ----

if __name__ == "__main__":
    main()
