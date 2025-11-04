import torch
import torch.func
from math import sqrt, prod
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
    t_range = torch.linspace(-0.01, 0.01, 100)

    for i in range(N_vec):
        random_vector = [torch.randn_like(param.data) for param in model.parameters()]
        loss_values = []
        for t in tqdm(t_range):
            loss = compute_loss_at_point(model, t, trained_params, random_vector, test_loader, criterion)
            loss_values.append(loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values

# I changed randn from autograd after2025/11/04
#layer-normalize
def analyze_loss_landgrad(model, test_loader, criterion, N_vec):
    device = next(model.parameters()).device
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-1, 1, 100)

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
        random_vector = [torch.randn_like(param.data) for param in model.parameters()]
        scales = [sqrt(prod(p.shape[1:])) if p.ndim > 1 else sqrt(p.shape[0]) for p in model.parameters()]
        scaled_random_vector = [d.div(scale) for (d, scale) in zip(random_vector, scales)]

        print(torch.sqrt(sum(p.norm()**2 for p in scaled_random_vector)).item())

        loss_values = []
        for t in tqdm(t_range):
            current_loss = compute_loss_at_point(model, t, trained_params, scaled_random_vector, test_loader, criterion)
            loss_values.append(current_loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values

#layer-normalize + norm-normalize
def analyze_loss_landgrad2(model, test_loader, criterion, N_vec):
    device = next(model.parameters()).device
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-10, 10, 100)

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
        random_vector = [torch.randn_like(param.data) for param in model.parameters()]
        scales = [sqrt(prod(p.shape[1:])) if p.ndim > 1 else sqrt(p.shape[0]) for p in model.parameters()]
        scaled_random_vector = [d.div(scale) for (d, scale) in zip(random_vector, scales)]

        grad_norm = torch.sqrt(sum(p.norm()**2 for p in scaled_random_vector))
        final_normalized_random_vector = [g / grad_norm for g in scaled_random_vector]

        print(torch.sqrt(sum(p.norm()**2 for p in final_normalized_random_vector)).item())

        loss_values = []
        for t in tqdm(t_range):
            current_loss = compute_loss_at_point(model, t, trained_params, final_normalized_random_vector, test_loader, criterion)
            loss_values.append(current_loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values


#ただ単に摂動ベクトルを正規化して用いる norm-normlize
def analyze_loss_landgrad_for_tiny(model, test_loader, criterion, N_vec):
    device = next(model.parameters()).device
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-1, 1, 100)

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
        #auto_gradをつかった摂動ベクトルは, トレーニング後のlosslandscapeの勾配から計算されるので, 0に非常に近くなってしまい, 意味をなさない可能性がある. そこで大きさを1に正規化する.
        grad_norm = torch.sqrt(sum(p.norm()**2 for p in grad_vector))
        normalized_grad_vector = [g / grad_norm for g in grad_vector]
        #摂動ベクトルの大きさが十分か一応確認 
        print(torch.sqrt(sum(p.norm()**2 for p in normalized_grad_vector)).item())

        loss_values = []
        for t in tqdm(t_range):
            current_loss = compute_loss_at_point(model, t, trained_params, normalized_grad_vector, test_loader, criterion)
            loss_values.append(current_loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values

#test用: 各layerごとの正規化を行った後に, 摂動ベクトルを1に正規化
#layer-normalize + norm-normalize
def analyze_loss_landgrad_for_tiny2(model, test_loader, criterion, N_vec):
    device = next(model.parameters()).device
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-1, 1, 100)

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
        scales = [sqrt(prod(p.shape[1:])) if p.ndim > 1 else sqrt(p.shape[0]) for p in model.parameters()]
        scaled_grad_vector = [d.div(scale) for (d, scale) in zip(grad_vector, scales)]
        #auto_gradをつかった摂動ベクトルは, トレーニング後のlosslandscapeの勾配から計算されるので, 0に非常に近くなってしまい, 意味をなさない可能性がある. そこで大きさを1に正規化する.
        grad_norm = torch.sqrt(sum(p.norm()**2 for p in scaled_grad_vector))
        final_normalized_grad_vector = [g / grad_norm for g in scaled_grad_vector]
        #摂動ベクトルの大きさが十分か一応確認 
        print(torch.sqrt(sum(p.norm()**2 for p in final_normalized_grad_vector)).item())

        loss_values = []
        for t in tqdm(t_range):
            current_loss = compute_loss_at_point(model, t, trained_params, final_normalized_grad_vector, test_loader, criterion)
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

#何かあったときのバックアップ用解析コード
def analyze_loss_landgrad(model, test_loader, criterion, N_vec):
    device = next(model.parameters()).device
    trained_params = {name: param.data.clone() for name, param in model.named_parameters()}
    all_loss_values = []
    t_range = torch.linspace(-10, 10, 100)

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
        scales = [sqrt(prod(p.shape[1:])) if p.ndim > 1 else sqrt(p.shape[0]) for p in model.parameters()]
        normalized_grad_vector = [d.div(scale) for (d, scale) in zip(grad_vector, scales)]

        print(torch.sqrt(sum(p.norm()**2 for p in normalized_grad_vector)).item())

        loss_values = []
        for t in tqdm(t_range):
            current_loss = compute_loss_at_point(model, t, trained_params, normalized_grad_vector, test_loader, criterion)
            loss_values.append(current_loss)
        all_loss_values.append(loss_values)

    return t_range, all_loss_values
"""