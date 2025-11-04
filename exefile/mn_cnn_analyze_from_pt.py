#学習済みモデルを使ってlosslandscapeを見る.
import sys
import os

# src/ ディレクトリをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# ファイル名でインポート
import torch
from train.mn_cnn_train_lrs import Trainer
from train.mn_cnn_train import train_model
from analysis.mn_cnn_losslandscape import analyze_loss_landscape, analyze_loss_landscape_multi, analyze_loss_landgrad
from visualization.mn_cnn_plots import plot_training_results 
import torch.nn as nn
from models.mn_cnn_overlr import MNISTcnn_ovlr
from data.mn_data_loader import get_mnist_loaders
from analysis.mn_cnn_losslandscape import analyze_loss_landgrad2, analyze_loss_landgrad_for_tiny2
from visualization.mn_cnn_plots import plot_training_results

def main():
    # モデル読み込み
    device = torch.device("cuda")
    model = MNISTcnn_ovlr()
    model_path = "models/mnist_cnn_lrs_1104.pt"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    _, test_loader = get_mnist_loaders(BATCH_SIZE=64)
    
    # 損失ランドスケープ分析
    criterion = nn.CrossEntropyLoss()
    t_range, loss_values = analyze_loss_landgrad2(model, test_loader, criterion, N_vec=5)

    # 可視化
    plot_training_results([], [], [], t_range, loss_values)

if __name__ == "__main__":
    main()