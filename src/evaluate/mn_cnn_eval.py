import torch

def evaluate_model(model, data_loader, device='cpu'):
    """
    指定されたデータローダーを使ってモデルの精度を評価します。

    Args:
        model (torch.nn.Module): 評価対象の学習済みモデル。
        data_loader (torch.utils.data.DataLoader): 評価に使用するデータローダー。
        device (str): 'cpu' または 'cuda'。モデルとデータを移動させるデバイス。

    Returns:
        float: データセットに対するモデルの正解率（%）。
    """
    model.eval()  # モデルを評価モードに設定
    model.to(device) # モデルを適切なデバイスへ移動

    correct = 0
    total = 0
    with torch.no_grad():  # 評価中は勾配計算を無効化
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 精度の計算
    accuracy = 100 * correct / total
    return accuracy