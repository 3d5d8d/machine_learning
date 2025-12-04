import torch
import torch.nn as nn
import sys
import os

# プロジェクトのルートディレクトリをPythonのパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# srcパッケージから必要なモジュールをインポート
from src.models.mn_cnn_overlr import MNISTcnn_ovlr # 正しいモデルをインポート
from src.data.mn_data_loader import get_mnist_loaders
from src.analysis.mn_cnn_losslandscape import analyze_hessian_spectrum
from src.analysis.mn_cnn_losslandscape import analyze_hessian_spectrum_ave
from src.visualization.mn_cnn_plots import plot_hessian_spectrum

def main():
    """
    学習済みモデルをロードし、ヘッセ行列の固有値スペクトルを計算・プロットする。
    """
    # --- 設定 ---
    MODEL_PATH = 'models/mnist_cnn_lrs_251104.pt'
    BATCH_SIZE = 64 #loss function計算に用いるデータのバッチサイズ. criterionに渡すときに使う.
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # --- 1. モデルのロード ---
    model = MNISTcnn_ovlr()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    print(f"モデルを '{MODEL_PATH}' からロードしました。")

    # --- 2. データローダーの準備 ---
    train_loader, _ = get_mnist_loaders(BATCH_SIZE=BATCH_SIZE)

    # --- 3. 損失関数の定義 ---
    criterion = nn.CrossEntropyLoss()

    # --- 4. ヘッセ行列の固有値スペクトルを計算 ---
    #eigenvalues = analyze_hessian_spectrum(model, train_loader, criterion, num_steps=500)
    eigenvalues = analyze_hessian_spectrum_ave(model, train_loader, criterion, num_steps=500, num_samples=5)

    # --- 5. 結果をプロット ---
    if eigenvalues is not None:
        plot_hessian_spectrum(eigenvalues)

if __name__ == '__main__':
    main()
