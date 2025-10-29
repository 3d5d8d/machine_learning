import sys
import os

# src/ ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ファイル名でインポート
import torch
from train.mn_cnn_train_lrs import Trainer
from train.mn_cnn_train import train_model
from analysis.mn_cnn_losslandscape import analyze_loss_landscape, analyze_loss_landscape_multi, analyze_loss_landgrad
from visualization.mn_cnn_plots import plot_training_results 
import torch.nn as nn

def main():
    # 学習実行
    #model, train_losses, train_accuracies, test_accuracies, test_loader = train_model() #when using mn_cnn_train.py
    device = torch.device("cuda")
    trainer = Trainer(epochs=200, lr=0.001, batch_size=64, device=device)
    model, train_losses, train_accuracies, test_accuracies, test_loader = trainer.run()
    
    # 損失ランドスケープ分析
    criterion = nn.CrossEntropyLoss()
    t_range, loss_values = analyze_loss_landgrad(model, test_loader, criterion, N_vec=5)

    # 可視化
    plot_training_results(train_losses, train_accuracies, test_accuracies, t_range, loss_values)
    
    print("全ての処理が完了しました！")

if __name__ == "__main__":
    main()