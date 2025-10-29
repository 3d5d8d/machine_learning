import sys
import os

# src/ ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ファイル名でインポート
import torch
from analysis.mn_cnn_losslandscape import analyze_loss_landscape, analyze_loss_landscape_multi, analyze_loss_landgrad
from visualization.mn_cnn_plots import plot_training_results 
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from evaluate.mn_cnn_eval import evaluate_model
from torchvision.models import ResNet34_Weights, resnet34
from tqdm import tqdm


def main():
    device = torch.device("cuda")

    #using pretrained model
    model = models.resnet34(weights = ResNet34_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    #preparing test_data, defining preprocess for resnet
    resnet_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # get_mnist_loaders は訓練/テスト両方を返すため、テストデータだけを使います。
    # また、transformを引数で渡せないので、ここでローダーを直接定義します。
    test_dataset = MNIST(root='./data', train=False, download=True, transform=resnet_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    #calculate loss
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Calculating Loss"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    test_loss = total_loss / len(test_loader)

    # evaluate accuracy
    test_accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()