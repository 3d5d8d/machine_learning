import torch
import torch.nn as nn
from src.data.cifar10_loader import get_cifar10_loaders
from src.models.cifar10_resnet import create_cifar10_resnet18
from src.analysis.mn_cnn_losslandscape import analyze_hessian_spectrum_ave4
from src.visualization.mn_cnn_plots import plot_hessian_spectrum

def main():
    device=torch.device("cuda")
    batch_size=64
    model = create_cifar10_resnet18(pretrained=False)
    #print(model) #check the layer name if needed
    target_layer = "layer1"
    
    model_path = "C:/Users/toria/MyProject/results/checkpoints/cifar10_resnet18_20260625-022105/last.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    train_loader, _ = get_cifar10_loaders(BATCH_SIZE=batch_size)
    criterion = nn.CrossEntropyLoss()

    eigenvalues, max_eigenvector = analyze_hessian_spectrum_ave4(
		model,
		train_loader,
		criterion,
		num_steps=10, 
		num_samples=20, #50000/64 can use all batch data.
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