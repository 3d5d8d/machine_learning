import matplotlib.pyplot as plt
import os

def plot_training_results(train_losses, train_accuracies, test_accuracies, t_range, loss_values):

    # Create results/figs directory if it doesn't exist
    os.makedirs('../results/figs', exist_ok=True)

    # 1. loss
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 3, 1)
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

    # 3. Loss Landscape (別画像)
    plt.figure(figsize=(8, 6))
    plt.plot(t_range.numpy(), loss_values, 'b-', linewidth=2)
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='θ (trained)')
    plt.xlabel('t')
    plt.ylabel('L(θ + tv)')
    plt.title('Loss Landscape')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../results/figs/loss_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Loss Landscape を loss_landscape.png に保存しました")
