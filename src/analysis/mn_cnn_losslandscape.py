import numpy as np
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

# get Eigenvalue spectral density by calculating Hessian
def _hessian_vector_product(loss, params, v):
    """
    Hessian-vector product (Hv) を計算するヘルパー関数。
    """
    # 1回目の微分: 損失の勾配 grad_L を計算
    grad_params = torch.autograd.grad(loss, params, create_graph=True)
    
    # grad_L とベクトル v の内積を計算
    # grad_L はタプルなので、各要素の内積を計算して合計する
    flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_params])
    v_dot_grad = torch.dot(flat_grad, v)
    
    # 2回目の微分: 内積の結果を再度パラメータで微分する
    # これが (v^T H) に相当する
    hvp = torch.autograd.grad(v_dot_grad, params, retain_graph=True)
    
    # タプル形式のhvpをフラットなベクトルに変換
    flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp])
    return flat_hvp

def analyze_hessian_spectrum(model, data_loader, criterion, num_steps):
    """
    ランチョス法を用いてヘッセ行列の固有値スペクトルを計算する。
    外部ライブラリに依存しない。

    Args:
        model (torch.nn.Module): 対象モデル
        data_loader (torch.utils.data.DataLoader): 訓練データローダー
        criterion (torch.nn.Module): 損失関数
        num_steps (int): ランチョス法の反復回数（計算する固有値の数に相当）

    Returns:
        np.ndarray: 計算されたヘッセ行列の固有値
    """
    device = next(model.parameters()).device
    model.eval()

    try:
        images, labels = next(iter(data_loader))
        images, labels = images.to(device), labels.to(device)
    except StopIteration:
        print("データローダーが空です。")
        return None

    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)

    outputs = model(images)
    loss = criterion(outputs, labels)

    q = torch.randn(num_params, device=device)
    q /= torch.norm(q)
    
    q_list = [torch.zeros_like(q), q]
    alpha_list = []
    beta_list = []

    print(f"ランチョス法を開始します (ステップ数: {num_steps})...")
    for k in tqdm(range(num_steps)):
        # ヘッセ行列ベクトル積 w_hat = Hq を計算
        w_hat = _hessian_vector_product(loss, params, q_list[-1])

        # alphaを計算
        alpha = torch.dot(w_hat, q_list[-1])
        alpha_list.append(alpha)

        # 直交化
        w = w_hat - alpha * q_list[-1]
        if k > 0:
            w = w - beta_list[-1] * q_list[-2]
        
        beta = torch.norm(w)
        
        if beta < 1e-8: #適宜変更
            print(f"反復 {k+1} でBetaがゼロに収束したため、早期終了します。")
            break
        
        # 最後の反復でない場合のみbetaをリストに追加
        if k < num_steps - 1:
            beta_list.append(beta)
            q_list.append(w / beta)

    # 実際に実行されたステップ数を取得
    actual_steps = len(alpha_list)
    if actual_steps == 0:
        print("計算を1ステップも実行できませんでした。")
        return None

    # 実行されたステップ数で三重対角行列 T を構築
    T = torch.zeros(actual_steps, actual_steps, device=device)
    alphas = torch.tensor(alpha_list, device=device)
    
    T.diagonal(0).copy_(alphas)
    if actual_steps > 1:
        # beta_listはalpha_listより1つ少ない
        betas = torch.tensor(beta_list, device=device)
        T.diagonal(-1).copy_(betas)
        T.diagonal(1).copy_(betas)
    
    #Tを対角化して, UθU^Tを得る
    eigenvalues = torch.linalg.eigh(T).eigenvalues
    
    print("ヘッセ行列の固有値計算が完了しました。")
    return eigenvalues.cpu().numpy()


def _Hessian_vector_product_for_ave(model, data_loader, criterion, params, v, num_samples):
    device = next(model.parameters()).device
    hvp_sum = torch.zeros_like(v)
    
    data_iter = iter(data_loader)
    for _ in range(num_samples):
        try:
            images, labels = next(data_iter)
            images, labels = images.to(device), labels.to(device)
        except StopIteration:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)
            images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
    
        # 1回目の微分
        grad_params = torch.autograd.grad(loss, params, create_graph=True)
        
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_params])
        v_dot_grad = torch.dot(flat_grad, v)
        
        # 2回目の微分
        hvp = torch.autograd.grad(v_dot_grad, params, retain_graph=True)
        flat_hvp = torch.cat([g.contiguous().view(-1) for g in hvp])
        
        hvp_sum += flat_hvp
    # 平均を返す
    return hvp_sum / num_samples

def analyze_hessian_spectrum_ave(model, data_loader, criterion, num_steps, num_samples):
    device = next(model.parameters()).device
    model.eval()

    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)

    q = torch.randn(num_params, device=device)
    q /= torch.norm(q)
    
    q_list = [torch.zeros_like(q), q]
    alpha_list = []
    beta_list = []

    print(f"ランチョス法を開始します (ステップ数: {num_steps}, バッチサンプル数: {num_samples})...")
    for k in tqdm(range(num_steps)):
        # 複数バッチで平均を取ったヘッセ行列ベクトル積を計算
        w_hat = _Hessian_vector_product_for_ave(model, data_loader, criterion, params, q_list[-1], num_samples)

        alpha = torch.dot(w_hat, q_list[-1])
        alpha_list.append(alpha)

        w = w_hat - alpha * q_list[-1]
        if k > 0:
            w = w - beta_list[-1] * q_list[-2]
        
        beta = torch.norm(w)
        
        if beta < 1e-8:
            print(f"反復 {k+1} でBetaがゼロに収束したため、早期終了します。")
            break
        
        if k < num_steps - 1:
            beta_list.append(beta)
            q_list.append(w / beta)

    actual_steps = len(alpha_list)
    if actual_steps == 0:
        print("計算を1ステップも実行できませんでした。")
        return None

    T = torch.zeros(actual_steps, actual_steps, device=device)
    alphas = torch.tensor(alpha_list, device=device)
    
    T.diagonal(0).copy_(alphas)
    if actual_steps > 1:
        betas = torch.tensor(beta_list, device=device)
        T.diagonal(-1).copy_(betas)
        T.diagonal(1).copy_(betas)

    eigenvalues = torch.linalg.eigh(T).eigenvalues
    
    print("ヘッセ行列の固有値計算が完了しました。")
    return eigenvalues.cpu().numpy()


def analyze_hessian_spectrum_ave2(model, data_loader, criterion, num_steps, num_samples):
    device = next(model.parameters()).device
    model.eval()

    params = list(model.parameters())
    num_params = sum(p.numel() for p in params)

    q = torch.randn(num_params, device=device)
    q /= torch.norm(q)
    
    q_list = [torch.zeros_like(q), q]
    alpha_list = []
    beta_list = []

    #
    for i in tqdm(range(num_steps)):
        w_hat = _Hessian_vector_product_for_ave(model, data_loader, criterion, params, q_list[-1], num_samples)

        alpha = torch.dot(w_hat, q_list[-1])
        alpha_list.append(alpha)

        w = w_hat - alpha * q_list[-1]
        if i > 0:
            w = w - beta_list[-1] * q_list[-2]
        
        beta = torch.norm(w)
        
        if beta < 1e-8:
            break
        
        if i < num_steps - 1:
            beta_list.append(beta)
            q_list.append(w / beta)

    actual_steps = len(alpha_list)
    if actual_steps == 0:
        return None, None

    T = torch.zeros(actual_steps, actual_steps, device=device)
    alphas = torch.tensor(alpha_list, device=device)
    T.diagonal(0).copy_(alphas)
    if actual_steps > 1:
        betas = torch.tensor(beta_list, device=device)
        T.diagonal(-1).copy_(betas)
        T.diagonal(1).copy_(betas)

    # 固有値と固有ベクトルを計算
    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    
    nodes = eigenvalues.cpu().numpy()
    # 固有ベクトルの第1成分の2乗が重み
    weights = (eigenvectors[0, :] ** 2).cpu().numpy()

    return nodes, weights

def compute_hessian_density(model, data_loader, criterion, num_steps, num_samples, n_vectors, sigma, t_range):
    """
    n_vectors (k): ランダムベクトルの試行回数
    sigma: ガウス核の幅
    t_range: 密度を計算するグリッド (numpy array)
    """
    total_density = np.zeros_like(t_range)
    
    print(f"スペクトル密度推定を開始 (試行回数 k={n_vectors})...")
    for _ in range(n_vectors):
        nodes, weights = analyze_hessian_spectrum_ave2(model, data_loader, criterion, num_steps, num_samples)
        
        if nodes is None: continue

        # ガウス関数 f(θ) の適用と加算
        for node, weight in zip(nodes, weights):
            total_density += weight * np.exp(-(t_range - node)**2 / (2 * sigma**2))

    # 正規化定数と試行回数kでの平均化
    normalization = 1.0 / (np.sqrt(2 * np.pi) * sigma * n_vectors)
    return total_density * normalization

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