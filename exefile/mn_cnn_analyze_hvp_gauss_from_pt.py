import torch
import torch.nn as nn
import sys
import os
import numpy as np

# プロジェクトのルートディレクトリをPythonのパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# srcパッケージから必要なモジュールをインポート
from src.models.mn_cnn_overlr import MNISTcnn_ovlr # 正しいモデルをインポート
from src.data.mn_data_loader import get_mnist_loaders
from src.analysis.mn_cnn_losslandscape import compute_hessian_density
from src.visualization.mn_cnn_plots import plot_hessian_density


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
    t_range = np.linspace(-1, 1, 1000)
    density = compute_hessian_density(
        model, train_loader, criterion, 
        num_steps=100, #ランチョス法の反復回数。どこまで次元落として射影するか
        num_samples=940, #batchの取得回数. batchsizeとの積が全データ数を超えないと正しく固有値の近似ができない
        n_vectors=10, #最後に期待値とるときの試行回数k 10
        sigma=0.1, 
        t_range=t_range
    )

    # --- 5. 結果をプロット ---
    plot_hessian_density(t_range, density)


if __name__ == '__main__':
    main()
