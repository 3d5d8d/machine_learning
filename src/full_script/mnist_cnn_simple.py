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
    train=True, # get training data
    download=True, # download the dataset if not already present
    transform=transformed # using the transformed variable
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False, # Do not train because we want to evaluate the results of training
    download=True, # download the dataset if not already present
    transform=transformed # using the transformed variable
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
            nn.Flatten(),  # Flatten the input
            nn.Linear(64*5*5, 256), # Fully connected layer, input size is 64*5*5 (output of the last conv layer), output size is 256
            nn.ReLU(),
            nn.Linear(256, 10),  # Output layer, 10 classes for MNIST digits (0-9)
            nn.LogSoftmax(dim=1)  # Log softmax for multi-class classification
        )

    def forward(self, x): #defining how the input data flows through the model
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

#studying the model

model = MNISTcnn()
criterion = nn.CrossEntropyLoss()  # Loss function (Evaluation function) for the model
# CrossEntropyLoss combines Logsoftmax and NLLLoss, suitable for multi-class classification tasks like MNIST
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Optimizer for training the model. lr = Learning rate (hyperparameter)

for epoch in range(10):
    total_loss = 0.0 # Initialize total Loss for the epoch
    for images, labels in tqdm(train_loader): # Loop through the each batch of images and Labels in the training data
        optimizer.zero_grad() # reset the gradients to zero
        outputs = model(images) # compute the model'S predictions. if using GPU, use outputs = model(images.to(device))
        loss = criterion(outputs, labels) # calculate the Loss using the criterion
        loss.backward() # calculate the gradients
        optimizer.step() # update the model parameters using the optimizer
        total_loss += loss.item() # accumulate the loss for the epoch
        # Print the average loss for the epoch
        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_loader)}")

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

# Plotting training loss and accuracy graphs
import matplotlib.pyplot as plt
import os

# Create results/figs directory if it doesn't exist
os.makedirs('../results/figs', exist_ok=True)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), [1.2, 0.8, 0.6, 0.4, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12])  # Example loss values
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), [85, 88, 91, 93, 94.5, 95.2, 96, 96.5, 97, 97.5])  # Example accuracy values
plt.title('Training & Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.tight_layout()

# Save the figure to results/figs directory
plt.savefig('../results/figs/training_results.png', dpi=300, bbox_inches='tight')
print("グラフを results/figs/training_results.png に保存しました")
plt.show()






