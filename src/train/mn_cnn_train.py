import torch
import torch.nn as nn
from tqdm import tqdm
from models.mn_cnn_simple import MNISTcnn
from data.mn_data_loader import get_mnist_loaders
from evaluate.mn_cnn_eval import evaluate_model

def train_model(epochs=30, lr=0.001, batch_size=64):
    train_loader, test_loader = get_mnist_loaders(batch_size)
    model = MNISTcnn()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        # training loop
        model.train()
        total_train = 0
        correct_train = 0
        total_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        # Evaluate on test data
        test_accuracy = evaluate_model(model, test_loader)
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')

    return model, train_losses, train_accuracies, test_accuracies, test_loader
