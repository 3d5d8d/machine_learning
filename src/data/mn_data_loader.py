import torch
from torchvision import datasets, transforms

def get_mnist_loaders(BATCH_SIZE=64):
    transformed = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transformed
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False, # Do not train because we want to evaluate the results of training
        download=True, 
        transform=transformed
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, test_loader
    