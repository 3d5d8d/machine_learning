import os 
import sys
import torch
import torch.nn as nn

from src.models.cifar10_resnet import create_cifar10_resnet18
from src.data.cifar10_loader import get_cifar10_loaders
from src.analysis.mn_cnn_losslandscape import analyze_loss_landgrad2
from src.visualization.mn_cnn_plots import plot_training_results

def main():
    device = torch.device("cuda")
    model = create_cifar10_resnet18(pretrained=False)
    model_path = "C:/Users/toria/MyProject/results/checkpoints/cifar10_resnet18_20260625-022105/last.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    _, test_loader = get_cifar10_loaders(
        BATCH_SIZE=64,
        augment=False,
        use_custom_aug=False,
        use_random_erasing=False
    )
    
    criterion = nn.CrossEntropyLoss()
    
    t_range, loss_values = analyze_loss_landgrad2(model, test_loader, criterion, N_vec=5)

    # 可視化
    plot_training_results([], [], [], t_range, loss_values)

if __name__ == "__main__":
    main()
    