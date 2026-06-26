from datetime import datetime
from src.models.cifar10_resnet import create_cifar10_resnet18
from src.train.cifar10_resnet_train import Cifar10Trainer
from src.data.cifar10_loader import get_cifar10_loaders
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

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
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = "cifar10_resnet18"

    log_dir = f"results/runs/{experiment_name}_{run_id}"
    save_dir = f"results/checkpoints/{experiment_name}_{run_id}"
    
    trainer = Cifar10Trainer(
        model = model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
        log_dir =log_dir,
        scheduler=scheduler,
        save_dir=save_dir,
        save_best=True,
        save_last=True,
    )
    trainer.run()
   

if __name__ == "__main__":
    main()

