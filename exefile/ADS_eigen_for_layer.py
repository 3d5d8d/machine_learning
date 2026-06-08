import os
import sys
import torch
import torch.nn as nn
import numpy as np
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.models.mn_cnn_overlr import MNISTcnn_ovlr
from src.data.mn_data_loader import get_mnist_loaders
from src.analysis.mn_cnn_losslandscape import analyze_hessian_spectrum_ave4
from src.visualization.mn_cnn_plots import plot_hessian_spectrum


def main():
    # 比較したい条件 (例：theta=0, theta=45)
    CONDITIONS = {
        "theta0": os.path.join(project_root, "models", "260605_theta0.pt"),
        "theta45": os.path.join(project_root, "models", "260605_theta45.pt"),
    }

    target_layers = ["layer1.0", "layer4.3"]

    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # データローダ
    train_loader, _ = get_mnist_loaders(BATCH_SIZE=64)
    criterion = nn.CrossEntropyLoss()

    results = []

    for cond_name, model_path in CONDITIONS.items():
        if not os.path.exists(model_path):
            print(f"モデルファイルが見つかりません: {model_path}  (スキップします)")
            continue

        # モデル初期化と読み込み
        model = MNISTcnn_ovlr()
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        for target_layer in target_layers:
            print(f"== Condition: {cond_name}, Target layer: {target_layer} ==")

            # analyze_hessian_spectrum_ave4 は全固有値配列と最大固有ベクトルを返す設計
            eigenvalues, max_eigenvector = analyze_hessian_spectrum_ave4(
                model,
                train_loader,
                criterion,
                num_steps=100,
                num_samples=940,
                target_layer=target_layer,
            )

            # 最大固有値（スカラー）
            max_eig = float(np.max(eigenvalues))

            # target_layer に属するパラメータだけを抽出して L2 ノルム二乗を計算
            target_params = [(name, p) for name, p in model.named_parameters() if target_layer in name]
            if len(target_params) == 0:
                print(f"ターゲットレイヤー '{target_layer}' のパラメータが見つかりません。")
                continue

            l2_sq = 0.0
            for name, p in target_params:
                l2_sq += float(torch.sum(p.data.cpu().double() ** 2))

            adaptive_sharpness = max_eig * l2_sq

            print(f"max_eig={max_eig:.6e}, ||w||^2={l2_sq:.6e}, AdaptiveSharpness={adaptive_sharpness:.6e}")

            # レイヤ内での固有ベクトルの寄与を表示
            current_idx = 0
            for name, param in target_params:
                num_params = param.numel()
                vec_part = max_eigenvector[current_idx: current_idx + num_params]
                contribution = float(torch.norm(vec_part).item() ** 2)
                print(f"  {name}: contribution={contribution:.6e}")
                current_idx += num_params

            results.append({
                "condition": cond_name,
                "layer": target_layer,
                "max_eigenvalue": max_eig,
                "l2_norm_sq": l2_sq,
                "adaptive_sharpness": adaptive_sharpness,
            })

            # restore requires_grad (安全対策)
            for _, p in model.named_parameters():
                p.requires_grad = True

    # 結果をCSVに保存 (results/csvdata フォルダ)
    out_dir = os.path.join(project_root, "results", "csvdata")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "adaptive_sharpness.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "layer", "max_eigenvalue", "l2_norm_sq", "adaptive_sharpness"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # optionally plot last computed spectrum scaled by ||w||^2
    if len(results) > 0 and 'eigenvalues' in locals():
        # use l2_sq from last iteration
        plot_hessian_spectrum(eigenvalues * l2_sq)
        print(f"結果を {out_path} に保存しました。")


if __name__ == "__main__":
    main()