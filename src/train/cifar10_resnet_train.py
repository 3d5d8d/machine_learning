from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from src.evaluate.mn_cnn_eval import evaluate_model

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=None):
    model.train()
    total_loss = 0
    correct_train = 0
    total_train = 0
    
    desc = "Training" if epoch is None else f"Training Epoch {epoch + 1}"
    
    for images, labels in tqdm(train_loader, desc=desc):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() #毎回次のバッチを取り出してきたら前の勾配計算の結果をゼロに
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    train_accuracy = 100*correct_train/total_train
    
    return avg_loss, train_accuracy



class Cifar10Trainer:
    def __init__(self, model, train_loader, test_loader, criterion, optimizer, device, epochs, log_dir, scheduler=None):
        self.model = model
        self.train_loader, self.test_loader = train_loader, test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir)
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.model.to(self.device)
        
    def run(self):
        try:
            for epoch in range(self.epochs):
                train_loss, train_acc = train_one_epoch(
                    model=self.model,
                    train_loader=self.train_loader,
                    criterion=self.criterion,
                    optimizer=self.optimizer,
                    device=self.device,
                    epoch=epoch,
                )
                
                test_acc = evaluate_model(
                    model=self.model,
                    data_loader=self.test_loader,
                    device=self.device,
                )
                
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)
                self.test_accuracies.append(test_acc)
                
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/test", test_acc, epoch)
                
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("Learning Rate", current_lr, epoch)
                print(
                    f"Epoch [{epoch + 1}/{self.epochs}] "
                    f"Loss: {train_loss:.4f} "
                    f"Train Acc: {train_acc:.2f}% "
                    f"Test Acc: {test_acc:.2f}% "
                    f"LR: {current_lr:.6f}"
                )
                
                if self.scheduler is not None:
                    self.scheduler.step()
        finally:
            self.writer.close()
        return self.model, self.train_losses, self.train_accuracies, self.test_accuracies