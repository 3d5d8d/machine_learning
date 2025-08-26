import sys
import os

# src/ ディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# 正しいファイル名でインポート
from train.mn_cnn_train import train_model
from analysis.mn_cnn_lossf import analyze_loss_landscape
from visualization.mn_cnn_plots import plot_training_results  # ファイル名修正
import torch.nn as nn

def main():
    # 学習実行
    model, train_losses, train_accuracies, test_accuracies, test_loader = train_model()
    
    # 損失ランドスケープ分析
    criterion = nn.CrossEntropyLoss()
    t_range, loss_values = analyze_loss_landscape(model, test_loader, criterion)
    
    # 可視化
    plot_training_results(train_losses, train_accuracies, test_accuracies, t_range, loss_values)
    
    print("全ての処理が完了しました！")

if __name__ == "__main__":
    main()