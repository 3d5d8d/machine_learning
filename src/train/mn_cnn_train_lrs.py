import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from models.mn_cnn_overlr import MNISTcnn_ovlr
from data.mn_data_loader import get_mnist_loaders
from evaluate.mn_cnn_eval import evaluate_model

class Trainer:
    def __init__(self, device, epochs=None, lr=None, batch_size=None):
        self.epochs = epochs
        self.device = device
        print(f"Using device: {self.device}")

        # 1. prepare TensorBoard SummaryWriter
        log_dir = f'results/runs/mnist_cnn_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard log directory: {log_dir}")

        # 2. データの準備
        self.train_loader, self.test_loader = get_mnist_loaders(batch_size)
        
        # 3. モデル、損失関数、オプティマイザ、スケジューラの準備
        self.model = MNISTcnn_ovlr().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=0) #select scheduler
        #self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.1) # if you want to use exp, uncomment this line and comment the above line

        # 4. 結果を記録するリスト
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

    def _train_epoch(self, epoch):
        """1エポック分の学習を実行する内部メソッド。"""
        self.model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for i, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # calculate gradient norm for monitoring (optional)
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar('Gradients/Total_Norm', total_norm, epoch * len(self.train_loader) + i)

            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct_train / total_train
        return epoch_loss, epoch_accuracy

    def run(self):
        """学習のメインループを実行する。"""
        try:
            for epoch in range(self.epochs):
                train_loss, train_accuracy = self._train_epoch(epoch)
                test_accuracy = evaluate_model(self.model, self.test_loader, self.device)
                
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_accuracy)
                self.test_accuracies.append(test_accuracy)
                
                # TensorBoardへの記録
                self.writer.add_scalar('Loss/train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                self.writer.add_scalar('Accuracy/test', test_accuracy, epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('Learning Rate', current_lr, epoch)

                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

                # スケジューラの更新
                self.scheduler.step()
        
        finally:
            self.writer.close()
            print("Finished Training. SummaryWriter closed.")
        
        # 学習結果を返す
        return self.model, self.train_losses, self.train_accuracies, self.test_accuracies, self.test_loader

