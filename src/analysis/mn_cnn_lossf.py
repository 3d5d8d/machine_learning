import torch
import torch.func
from tqdm import tqdm

def compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion):
    # creating dictionary of perturbed parameters, and updating them
    perturbed_params = {}
    for (name, param), rand_vec in zip(model.named_parameters(), random_vector):
        perturbed_params[name] = param + t * rand_vec

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            # apply parameters temporarily by using functional_call
            outputs = torch.func.functional_call(model, perturbed_params, images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(test_loader)

def analyze_loss_landscape(model, test_loader, criterion):
    # 訓練済みパラメータを辞書形式で保存
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}

    # create random perturbation vectors
    random_vector = [torch.randn_like(param.data) for param in model.parameters()]

    t_range = torch.linspace(-0.01, 0.01, 100)
    loss_values = []
    for t in tqdm(t_range):
        loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion)
        loss_values.append(loss)

    return t_range, loss_values


# 元のコード（コメントアウト）
"""
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

    t_range = torch.linspace(-0.05, 0.05, 50)
    loss_values = []
    for t in tqdm(t_range): # if no need for progress bar, use: for t in t_range:
        loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion) # t is a tensor,if you like explicitly, use t.item() to get the value
        loss_values.append(loss)

    for i, param in enumerate(model.parameters()):
        param.data = trained_params[i]

    return t_range, loss_values
"""