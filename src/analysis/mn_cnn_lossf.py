import torch
from tqdm import tqdm

def compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion):
    for i, param in enumerate(model.parameters()):
        param.data = trained_params[i]+t*random_vector[i]

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(test_loader)

def analyze_loss_landscape(model, test_loader, criterion):
    trained_params = [param.data.clone() for param in model.parameters()]

    random_vector = [torch.randn_like(param.data) for param in model.parameters()]

    t_range = torch.linspace(-1, -1, 50)
    loss_values = []
    for t in tqdm(t_range): # if no need for progress bar, use: for t in t_range:
        loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion) # t is a tensor,if you like explicitly, use t.item() to get the value
        loss_values.append(loss)

    for i, param in enumerate(model.parameters()):
        param.data = trained_params[i]

    return t_range, loss_values