import os
import sys

import torch
import torch.nn as nn

# プロジェクトのルートディレクトリを Python のパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.analysis.mn_cnn_losslandscape import analyze_hessian_spectrum_ave4
from src.data.mn_data_loader import get_mnist_loaders
from src.models.mn_cnn_overlr import MNISTcnn_ovlr
from src.visualization.mn_cnn_plots import plot_hessian_spectrum


def main():
	"""学習済みモデルをロードし、指定レイヤー固定版のヘッセ固有値を計算する。"""
	model_path = os.path.join("models", "mnist_cnn_lrs_create_mn_aug2_260318.pt")
	batch_size = 64
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# None で全層、文字列指定でその名前を含むパラメータのみを解析
	target_layer = "layer4.3"

	print(f"Using device: {device}")

	model = MNISTcnn_ovlr()
	model.load_state_dict(torch.load(model_path, map_location=device))
	model.to(device)
	print(f"モデルを '{model_path}' からロードしました。")

	train_loader, _ = get_mnist_loaders(BATCH_SIZE=batch_size)
	criterion = nn.CrossEntropyLoss()

	eigenvalues, max_eigenvector = analyze_hessian_spectrum_ave4(
		model,
		train_loader,
		criterion,
		num_steps=100,
		num_samples=940,
		target_layer=target_layer,
	)

	if eigenvalues is None:
		print("固有値計算に失敗したため終了します。")
		return

	plot_hessian_spectrum(eigenvalues)

	if target_layer is None:
		target_params = [(name, p) for name, p in model.named_parameters()]
	else:
		target_params = [(name, p) for name, p in model.named_parameters() if target_layer in name]

	if not target_params:
		print(f"寄与率表示対象がありません: target_layer={target_layer}")
		return

	current_idx = 0
	print("最大固有ベクトルのレイヤー別寄与率:")
	for name, param in target_params:
		num_params = param.numel()
		contribution = torch.norm(max_eigenvector[current_idx: current_idx + num_params]).item() ** 2
		print(f"{name:<30}: {contribution:.6f}")
		current_idx += num_params


if __name__ == "__main__":
	main()
