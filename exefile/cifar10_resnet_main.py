from datetime import datetime
from src.models.cifar10_resnet import create_cifar10_resnet18
from src.train.cifar10_resnet_train import Cifar10Trainer
from src.data.cifar10_loader import get_cifar10_loaders
import torch
import torch.nn as nn
from torch.optim import Adam

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 64
    lr = 0.001
    epochs = 200
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE=batch_size)
    model = create_cifar10_resnet18(pretrained=True).to(device) #images, labels is on GPU, so model should be on GPU as follow.
    criterion = nn.CrossEntropyLoss()
    params =model.parameters()
    optimizer = Adam(params, lr, eps=1e-08, weight_decay=0)
    
    trainer = Cifar10Trainer(
        model = model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        log_dir = f"results/runs/cifar10_resnet18_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        scheduler=None,
    )
    trainer.run()
   

if __name__ == "__main__":
    main()

