import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .transforms import AddGaussianNoise
#Using torch library, we don't have to set complicated settings, like one-hot, faltten(Basicaly, this dont need for CNN)

#This is 
def create_mnist_augmentation(noise_level=0.1):
    return transforms.Compose([
        #Geometric augmentations
        transforms.RandomAffine(
            degrees=15, #Random rotation[-degrees, degrees]
            translate=(0.1, 0.1),#random translation up to 10％
            scale=(0.8, 1.2) #random scaling[0.8, 1.2]
        ),
        #Photometric augmentations
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3
        ),
        #transform into tensor
        transforms.ToTensor(),
        
        #additional noise
        AddGaussianNoise(std=noise_level),
        
        #Random erasing (cutout) -good for MNIST
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0 #Fill with black(0) for MNIST
        )
    ])
    
    
#This is the main body of the data loader.
def get_mnist_loaders(BATCH_SIZE=64):
    
    train_transformed = create_mnist_augmentation() #If I want to use a clear image for training, put it into transforms.Compose([transforms.ToTensor()])
    test_transformed = transforms.Compose([transforms.ToTensor()]) #test data should be original for evaluating model's spec.

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transformed
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False, # Do not train because we want to evaluate the results of training
        download=True, 
        transform=test_transformed
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
    