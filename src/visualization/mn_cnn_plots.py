import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_results(train_losses, train_accuracies, test_accuracies, t_range, loss_values):

    # Create results/figs directory if it doesn't exist
    os.makedirs('../results/figs', exist_ok=True)

    # 1. loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figs/training_loss.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training Loss を training_loss.png に保存しました")

    # 2. Training and Test Accuracy (別画像)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(range(1, len(test_accuracies)+1), test_accuracies, 'r-', label='Test Accuracy', linewidth=2)
    plt.title('Training & Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figs/training_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training & Test Accuracy を training_accuracy.png に保存しました")

    # 3. Loss Landscape (adapting multivec)
    plt.figure(figsize=(8, 6))

    # if Judging solo or multiple(list)
    if loss_values and isinstance(loss_values[0], list):
        plt.title('Multiple Loss Landscapes')
        for i, single_loss_values in enumerate(loss_values):
            plt.plot(t_range.numpy(), single_loss_values, linewidth=2, label=f'Vector {i+1}')
    else:
        # 従来通りの単一プロット
        plt.title('Loss Landscape')
        plt.plot(t_range.numpy(), loss_values, 'b-', linewidth=2)

    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='θ (trained)')
    plt.xlabel('t')
    plt.ylabel('L(θ + tv)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figs/loss_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Loss Landscape を loss_landscape.png に保存しました")


def plot_hessian_spectrum(eigenvalues, save_dir='../results/figs'):
    """
    ヘッセ行列の固有値スペクトル（分布）をプロットします。

    Args:
        eigenvalues (np.ndarray): 計算された固有値の配列。
        save_dir (str): プロットを保存するディレクトリ。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues, bins=500, density=True)
    plt.title('Hessian Eigenvalue Spectrum')
    plt.xlabel('Eigenvalue (λ)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'hessian_spectrum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Hessian Spectrum plot を {save_path} に保存しました")

def plot_hessian_density(t_range, density, save_dir='../results/figs'):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.plot(t_range, density, linewidth=2)
    plt.title('Hessian Spectral Density')
    plt.xlabel('Eigenvalue (λ)')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, 'hessian_density.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Hessian Density plot を {save_path} に保存しました")