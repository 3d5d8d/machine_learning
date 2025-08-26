import torch
import torch.nn as nn
from tqdm import tqdm
from models.mn_cnn_simple import MNISTcnn
from data.mn_data_loader import get_mnist_loaders   #File name = module name

def train_model(epochs=10, lr=0.001, batch_size=64):
    train_loader, test_loader = get_mnist_loaders(batch_size)
    model = MNISTcnn()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        #training loop
        model.train()
        total_train = 0
        correct_train = 0
        total_loss = 0

        for images, labels in tqdm(train_loader):
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

        # --- テストデータで評価 ---
        model.eval() # 評価モード
        correct_test = 0
        total_test = 0
        with torch.no_grad(): # 勾配計算は不要
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        epoch_test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(epoch_test_accuracy)

        print(f"Epoch:{epoch+1}, Loss:{epoch_loss:.4f}, Train Acc:{epoch_accuracy:.2f}%, Test Acc:{epoch_test_accuracy:.2f}%")

    return model, train_losses, train_accuracies, test_accuracies, test_loader

if __name__ == "__main__":
    model, losses, accuracies, test_accuracies, test_loader = train_model()
    torch.save(model.state_dict(), "../models/mnist_cnn_trained.pt")
