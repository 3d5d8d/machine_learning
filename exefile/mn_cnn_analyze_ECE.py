# 学習済みモデルを使ってExpected Calibration Error (ECE)を計算する
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from models.mn_cnn_overlr import MNISTcnn_ovlr
from data.mn_data_loader import get_mnist_loaders

def calc_ece(preds_probs, true_labels, n_bins=10):
        
    #making interval Im=(m-1/M, m/M]
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    confidences = []
    accuracies = []
    bin_counts = []
    
    total_samples = len(preds_probs)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        #First bin → lower <= p <= upper, otherwise → lower < p <= upper
        if i == 0:
            in_bin = (preds_probs >= bin_lower) & (preds_probs <= bin_upper)
        else:
            in_bin = (preds_probs > bin_lower) & (preds_probs <= bin_upper)
            
        Bm_count = np.sum(in_bin) #A number of sumple in the bin.=|Bm|
        bin_counts.append(Bm_count)
        
        if Bm_count > 0: 
            acc_Bm = np.mean(true_labels[in_bin])
            conf_Bm = np.mean(preds_probs[in_bin])
            #(3) difinition of Expected Calibration Error
            ece += (Bm_count / total_samples) * np.abs(acc_Bm - conf_Bm)
            #save
            confidences.append(conf_Bm)
            accuracies.append(acc_Bm)
        else:
            confidences.append(0)
            accuracies.append(0)
            
    return ece, (confidences, accuracies, bin_counts)


def get_predictions(model, data_loader, device):
    model.eval()
    model.to(device)
    
    all_confidences = []
    all_correctness = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # モデル出力 (LogSoftmaxの出力)
            log_probs = model(images)
            
            # 確率に変換 (LogSoftmax -> Softmax)
            probs = torch.exp(log_probs)
            
            # 最大確率とそのクラスを取得
            conf, preds = torch.max(probs, dim=1)
            
            # 正解かどうか判定 (1 or 0)
            correct = (preds == labels).float()
            
            all_confidences.append(conf.cpu().numpy())
            all_correctness.append(correct.cpu().numpy())
    
    # 全サンプルを結合
    confidences = np.concatenate(all_confidences)
    correctness = np.concatenate(all_correctness)
    
    return confidences, correctness


def plot_reliability_diagram(confidences, accuracies, bin_counts, ece_score, n_bins=10, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    bin_centers = np.linspace(1/(2*n_bins), 1 - 1/(2*n_bins), n_bins)
    mask = np.array(bin_counts) > 0
    plt.bar(np.array(bin_centers)[mask], np.array(accuracies)[mask], 
            width=1.0/n_bins, edgecolor='black', color='blue', alpha=0.8, label='Outputs (Accuracy)')
    
    for i in range(n_bins):
        if bin_counts[i] > 0:
            acc = accuracies[i]
            conf = confidences[i]
            plt.bar(bin_centers[i], height=np.abs(acc - conf), bottom=min(acc, conf),
                    width=1.0/n_bins, color='red', alpha=0.3, hatch='//', edgecolor='red', label='Gap' if i==0 else "")

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'Reliability Diagram (ECE = {ece_score:.4f})')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = MNISTcnn_ovlr()
    model_path = "models/mnist_cnn_lrs_erasevalue0_changed_260209.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {model_path}")
    
    # データローダー取得
    _, test_loader = get_mnist_loaders(BATCH_SIZE=64)
    
    # 予測確率と正解フラグを取得
    confidences, correctness = get_predictions(model, test_loader, device)
    print(f"Total samples: {len(confidences)}")
    print(f"Accuracy: {np.mean(correctness) * 100:.2f}%")
    
    # ECE計算
    n_bins = 10
    ece_score, (bin_confs, bin_accs, bin_counts) = calc_ece(confidences, correctness, n_bins=n_bins)
    print(f"ECE Score: {ece_score:.4f}")
    
    # 信頼度図を描画
    save_path = "results/figs/ece_reliability_diagram.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plot_reliability_diagram(bin_confs, bin_accs, bin_counts, ece_score, n_bins=n_bins, save_path=save_path)


if __name__ == "__main__":
    main()
