import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .transforms import AddGaussianNoise
#Using torch library, we don't have to set complicated settings, like one-hot, faltten(Basicaly, this dont need for CNN)

import random

#This is for setting augmentation in training sequence. 
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
        
        #Random erasing (cutout) -good for MNIST
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0 #Fill with black(0) for MNIST
        ),
        #additional noise
        AddGaussianNoise(std=noise_level)
    ])

#This is function for those who want to set several augmentations automatically(randomly).
class Randomaugment:
    def __call__(self, img):
        choice = random.choices(["rotate", "translate", "scale"])
        magnitude = random.random()
        if choice =="rotate":
            Max_angle=5.0 #The default value = 15
            angle = Max_angle*magnitude*random.choice([-1, 1])
            return transforms.functional.rotate(img, angle)
        elif choice == "translate":
            max_tx = 0.5 #The default value = 0.1
            max_ty = 0.5
            tx = max_tx*img.size[0]*random.choice([-1, 1])
            ty = max_ty*img.size[1]*random.choice([-1, 1])
            return transforms.functional.affine(img, angle=0, translate=[tx, ty], scale=1.0, shear=0)
        else:
            max_scale = 0.5 #The default value = 0.2 
            scale = 1.0 + (max_scale*magnitude*random.choice([-1, 1]))
            return transforms.functional.affine(img, angle=0, translate=[0, 0], scale=scale, shear=0)

def create_mnist_augmentation2(noise_level=0.1):
    return transforms.Compose([
        Randomaugment(),
        #transform into tensor
        transforms.ToTensor(),
        #Random erasing (cutout) -good for MNIST
        transforms.RandomErasing(
            p=0.3,
            scale=(0.02, 0.1),
            ratio=(0.3, 3.3),
            value=0 #Fill with black(0) for MNIST
        ),
        #additional noise. After random erasing!!
        AddGaussianNoise(std=noise_level)
    ])
        
#This is the main body of the data loader.
def get_mnist_loaders(BATCH_SIZE=64):
    
    train_transformed = create_mnist_augmentation2() #If I want to use a clear image for training, put it into transforms.Compose([transforms.ToTensor()])
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
    