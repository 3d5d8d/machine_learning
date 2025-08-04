import torch
import torch.nn as nn

class MNISTcnn(nn.Module):
    def __init__(self):
        super().__init__()
        # defining the layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),    # 1 input channel, 32 output channels, filter size 3x3, stride size 1
            nn.ReLU(),                 # ReLU activation function
            nn.MaxPool2d(2, 2),           # Max pooling with a 2x2 window
            nn.Dropout(0.1)             # Dropout layer with a probability
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 256), # Fully connected layer, input size is 64*5*5 (output of the last conv layer), output size is 256
            nn.ReLU(),
            nn.Linear(256, 10),  # Output layer, 10 classes for MNIST digits (0-9)
            nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x