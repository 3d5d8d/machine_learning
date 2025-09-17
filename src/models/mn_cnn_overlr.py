import torch
import torch.nn as nn

class MNISTcnn_ovlr(nn.Module):
    def __init__(self):
        super().__init__()
        # defining the layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1),    # padding=1を追加
            nn.BatchNorm2d(32),
            nn.ReLU(),                 # ReLU activation function
            nn.MaxPool2d(2, 2),           # Max pooling with a 2x2 window
            nn.Dropout(0.3)             # Dropout layer with a probability
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),   # padding=1を追加
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )


        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),  # padding=1を追加
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )

        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*3*3, 256), # paddingありの場合は3x3になる
            nn.ReLU(),
            nn.Linear(256, 10),  # Output layer, 10 classes for MNIST digits (0-9)
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x