# this program based on the code from src/mnist_cnn_simple.py
# I add the function to compute loss parameters 

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# setting datasets
BATCH_SIZE = 64

# transform for MNIST dataset to tensor and normalization
transformed = transforms.Compose([transforms.ToTensor()])
# For ToTensor, the range is automatically scaled to [0, 1] ←grayscale images

# getting MNIST dataset
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

# making DataLoader for training dataset
# DataLoader is a utility to load data in batches
train_loader = torch.utils.data.DataLoader(
    train_dataset,              # datasets used in this program
    batch_size = BATCH_SIZE,    # A number of images in each batch
    shuffle = True,             # Shuffle the data (for stable training)
)

# making DataLoader for test dataset
test_loader = torch.utils.data.DataLoader(
    test_dataset,               # test data (unseen during training)
    batch_size=BATCH_SIZE,      # same batch size as training
    shuffle=False               # No need to shuffle for evaluation
)

# checking the content and type of the dataset. if needed remove the "#"
#print(train_dataset[0][0].shape) #should be torch.Size([1, 28, 28]) i.e. it is image data
#print(train_dataset[0][1]) #should be a label (0-9)
#print(type(train_dataset[0][0]))  # Should be a torch.Tensor
#print(type(train_dataset[0][1]))  # Should be an int (label) first[] is order

# making a CNN model
class MNISTcnn(nn.Module): # inherit from nn.Module for defining a class
    def __init__(self):
        super().__init__() # call the parent class constructor
        # defining the Layers
        self.layer1 = nn.Sequential( # applying a module in sequence, so simplifying model building
            nn.Conv2d(1, 32, 3, 1),    # 1 input channel, 32 output channels, filter size 3x3, stride size 1
            nn.ReLU(),                 # ReLU activation function
            nn.MaxPool2d(2, 2),           # Max pooling with a 2x2 window
            nn.Dropout(0.1) # Prevent overfitting by randomly setting 10% of the input to zero
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1)
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*5*5, 256), # Fully connected layer, input size is 64*5*5 (output of the last conv layer), output size is 256
            nn.ReLU(),
            nn.Linear(256, 10),  # Output layer, 10 classes for MNIST digits (0-9)
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x): #defining how the input data flows through the model
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#training the model

model = MNISTcnn()
criterion = nn.CrossEntropyLoss()  # Loss function (Evaluation function) for the model

# CrossEntropyLoss combines Logsoftmax and NLLLoss, suitable for multi-class classification tasks like MNIST
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for training the model. lr = Learning rate (hyperparameter)

train_losses = [] # List to store training losses for plotting later
train_accuracies = [] # List to store training accuracies for plotting later

for epoch in range(10):
    model.train()
    total_train = 0  # Initialize total number of training samples
    correct_train = 0  # Initialize number of correct predictions in training
    total_loss = 0.0 # Initialize total Loss for the epoch
    for images, labels in tqdm(train_loader): # Loop through the each batch of images and Labels in the training data
        optimizer.zero_grad()
        outputs = model(images) # compute the model'S predictions. if using GPU, use outputs = model(images.to(device))
        loss = criterion(outputs, labels) # calculate the Loss using the criterion
        loss.backward() # calculate the gradients
        optimizer.step() # update the model parameters using the optimizer
        total_loss += loss.item() # accumulate the loss for the epoch

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # memorize the average loss and accuracy for the epoch
    epoch_loss = total_loss / len(train_loader)  # Average loss for the epoch
    epoch_accuracy = 100 * correct_train / total_train  # Average accuracy for the epoch
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    # Print the average loss for the epoch
    print(f"Epoch: {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}%")

# Evaluate the model on the test dataset
correct = 0
total = 0
model.eval() #set the model to evaluation mode
with torch.no_grad(): # No need to calculate gradients during testing
    for images, labels in test_loader: # it is because test_loader has two variables, images and labels
        outputs = model(images) 
        _, predicted = torch.max(outputs.data, 1) # we want only the index of the max value in the output of neuron
        total += labels.size(0) # count already calculated in the test_loader
        correct += (predicted == labels).sum().item() #count the number of correct predictions

print(f'Accuracy: {100 * correct / total}%')

# We tweak the code to compute the loss parameters after training
# Getting the trained model parameters

trained_params = []
for param in model.parameters():
    trained_params.append(param.data.clone()) # create a copy of the parameter data because we don't want to modify the original parameters
# param.data is a tensor that contains the actual values of the parameters (weights and biases) after training, and ".data" is provided by Pytorch

#if needed, you can print the number of parameters and their shapes

#print(f"Number of parameters:{len(trained_params)}")
#print(f"Shapes of each parameter: {[p.shape for p in trained_params]}")

# creating a random vector with the same shape as the trained parameters
random_vector =[]
for param in model.parameters():
    v_elements = torch.randn_like(param.data) # create a random tensor with the same shape as the original parameter
    random_vector.append(v_elements)

# defining L(theta + t*v) theta= trained_params, v = random_vector
def compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion):
    # change the L's variables from model.parameters (=theta) to trained_params + t*random_vector
    for i, param in enumerate(model.parameters()):
        param.data = trained_params[i] + t * random_vector[i]

    # calculate the loss on the test dataset
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels) # criterion's variable is param.data
            total_loss += loss.item()

    return total_loss/len(test_loader)

# t in the range from -1 to 1
t_range = torch.linspace(-1, 1, steps=50) # t is a tensor with 50 values from -1 to 1
loss_values = []
for t in tqdm(t_range): # if no need for progress bar, use: for t in t_range:
    loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion) # t is a tensor,if you like explicitly, use t.item() to get the value
    loss_values.append(loss)

#============option part for plotting=================================
# Plotting training loss and accuracy graphs
import matplotlib.pyplot as plt
import os

# Create results/figs directory if it doesn't exist
os.makedirs('../results/figs', exist_ok=True)

plt.figure(figsize=(18, 4))  # 幅を広げる

# 1. loss
plt.subplot(1, 3, 1)
plt.plot(range(1, 11), train_losses, 'b-', linewidth=2)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 2. accuracy
plt.subplot(1, 3, 2)
plt.plot(range(1, 11), train_accuracies, 'b-', linewidth=2)
plt.title('Training & Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

# 3. loss landscape
plt.subplot(1, 3, 3)
plt.plot(t_range.numpy(), loss_values, 'b-', linewidth=2)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.7, label='θ (trained)')
plt.xlabel('t')
plt.ylabel('L(θ + tv)')
plt.title('Loss Landscape')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/figs/training_results.png', dpi=300, bbox_inches='tight')
print("グラフを results/figs/training_results.png に保存しました")
plt.show()

#============end of option part for plotting=================================

# returning the model to its original parameters
# This is optional, if you want to use the model after computing the loss landscape
for i, param in enumerate(model.parameters()):
    param.data = trained_params[i]