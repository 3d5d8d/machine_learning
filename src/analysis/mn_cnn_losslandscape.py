import torch
import torch.func
from tqdm import tqdm

def compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion):
    # creating dictionary of perturbed parameters, and updating them
    device = next(model.parameters()).device
    perturbed_params = {}
    for (name, param), rand_vec in zip(model.named_parameters(), random_vector):
        perturbed_params[name] = param + t * rand_vec

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
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

    # 各層の方向ベクトルを、対応する重み（パラメータ）のノルムで正規化する
    # これが簡易版の Filter-wise Normalization です
    for i, param in enumerate(model.parameters()):
        # ゼロ除算を避ける
        if torch.norm(param.data) > 1e-8 and torch.norm(random_vector[i]) > 1e-8:
            random_vector[i] = random_vector[i] * (torch.norm(param.data) / torch.norm(random_vector[i]))
    
    # tの範囲をより適切に修正
    t_range = torch.linspace(-0.01, 0.01, 100)
    loss_values = []
    for t in tqdm(t_range):
        loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion)
        loss_values.append(loss)

    return t_range, loss_values


def analyze_loss_landscape_multi(model, test_loader, criterion, N_vec=5):
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-0.5, 0.5, 100)

    for i in range(N_vec):
        random_vector = [torch.randn_like(param.data) for param in model.parameters()]
        loss_values = []
        for t in tqdm(t_range):
            loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion)
            loss_values.append(loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values


def analyze_loss_landgrad(model, test_loader, criterion, N_vec):
    device = next(model.parameters()).device
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-0.01, 0.01, 100)

    #ここから↓
    data_iter = iter(test_loader)
    for i in range(N_vec):

        try:
            images, labels = next(data_iter)
            #image, label = images[0].unsqueeze(0), labels[0].unsqueeze(0) # バッチから1つだけ取り出す
            images, labels = images.to(device), labels.to(device)
        except StopIteration:
            print("データローダーの終端に達しました。")
            break
        
        model.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        grad_vector = torch.autograd.grad(loss, model.parameters())
    #↑ここまで追加
        loss_values = []
        for t in tqdm(t_range):
            current_loss = compute_loss_at_point(model, t, trained_params, grad_vector, test_loader, criterion)
            loss_values.append(current_loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values


#old version
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