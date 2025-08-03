from sklearn.datasets import fetch_openml # scikit-learnのデータ取得ライブラリ
import numpy as np
from sklearn.model_selection import train_test_split # Testデータ生成用ライブラリ 
import torch # PyTorch
import torch.nn as nn # ニューラルネットワーク生成用ライブラリ
import torch.nn.functional as F # ニューラルネットワーク用の関数
import torch.optim as optim # 最適化関数のライブラリ
import matplotlib.pyplot as plt  # 追加


# Xに画像データ、tに対応する正解ラベル(0~9)が取得されます。
X, t = fetch_openml('mnist_784', version=1, return_X_y=True)
 
# 取得したデータを確認
print(X.shape, t.shape) # (70000, 784) (70000,)
print(type(X), type(t)) # <class 'numpy.ndarray'> <class 'numpy.ndarray'>

# データを28x28の形に変換してnumpy arrayにする  # 追加
X = X.values.reshape(-1, 28, 28)  # 追加
t = t.astype(int).values  # 追加
train_X, test_X, train_t, test_t = train_test_split(X, t, test_size=0.2, random_state=42)  # 追加

# creating CNN model
class Model(nn.Module):
   # 使用するニューラルネットの層を定義していきます definrg the layers of the neural network
    def __init__(self): # 層の定義. defining the layers
        super().__init__() # 親クラスの初期化を行う. initializing the parent class
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1) 
         # 1チャンネルの入力画像を10チャンネルに変換する畳み込み層. convolution layer to convert 1-channel input image to 10-channel# はじめの入力は1チャンネル、出力はフィルター数で10チャンネルの畳み込み層. convolution layer with 1 input channel and 10 output channels
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3, padding=1) #10種類の特徴（線, 横線, エッジとか色々）から20種類の特徴を検出する畳み込み層. convolution layer to detect 20 types of features from 10 types of features
        self.bn1 = nn.BatchNorm2d(num_features=20) #学習の安定化のためにバッチ正規化を行う. normalization for stable learning
        self.conv3 = nn.Conv2d(20, 20, kernel_size=3, padding=1) # 特徴量を精緻化するため畳み込み層. convolution layer to detect features
        self.conv4 = nn.Conv2d(20, 40, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=40)
        self.conv5 = nn.Conv2d(40, 40, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(40, 40, kernel_size=3, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=40)
 
        self.fc1 = nn.Linear(40 * 3 * 3, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 10)
 
    # __init__で定義した層を用いて、順伝播の処理を記載していきます。実際の計算処理を行うforwardメソッドを定義します。
    def forward(self, x):
        # import pdb; pdb.set_trace()
        x = F.relu(self.conv1(x)) # 入力画像に対して畳み込み層を適用し、ReLU活性化関数を適用します. apply convolution layer and ReLU activation function
        x = F.max_pool2d(F.relu(self.bn1(self.conv2(x))), (2,2)) #1+3-1で3×3の受容野を持つ畳み込み層を適用し、バッチ正規化とReLU活性化関数を適用します. apply convolution layer with 3x3 receptive field, batch normalization and ReLU activation function
        #pooling層であるMaxPoolにはパラメータ, 学習する重みがないので定義不要. ただ2×2で最大値をとるだけの処理. パラメータのある畳み込み層は定義で重みとバイアスが必要
        x = F.relu(self.conv3(x)) #活性化関数としてはReLUの使用が一般的. apply ReLU activation function ReLU(x) = max(0, x)
        x = F.max_pool2d(F.relu(self.bn2(self.conv4(x))), (2,2)) #3+(3-1)=5 で5×5の受容野を持つ畳み込み層を適用し、バッチ正規化とReLU活性化関数を適用します. apply convolution layer with 5x5 receptive field, batch normalization and ReLU activation function
        x = F.relu(self.conv5(x))
        x = F.relu(self.bn3(self.conv6(x))) #9×9の受容野を持つ畳み込み層を適用し、バッチ正規化とReLU活性化関数を適用します. apply convolution layer with 9x9 receptive field, batch normalization and ReLU activation function
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
    def num_flat_features(self, x): #全結合層に入力するために、畳み込み層の出力を1次元に変換するメソッド. method to convert the output of convolution layer to 1D for fully connected layer
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features 

# deviceの定義  # 追加
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 追加

# モデルのインスタンス化  # 追加
model = Model().to(device)  # 追加

# 損失関数の定義
criterion = nn.CrossEntropyLoss()
 
# 評価関数の定義
# 最適化関数(optimizer)の定義
# 第一引数にoptによって最適化するパラメータ群を指定する。モデルのパラメータ群になるはず。
lr = 1e-3
opt = optim.Adam(model.parameters(), lr=lr)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
 
        super(MyDataset, self).__init__()
        self.dataset = dataset
 
    # データセットの数
    def __len__(self):
        return len(self.dataset[0])
 
    # DataLoaderに呼ばれた際に取得するデータセット
    def __getitem__(self, idx):
 
        # idx単位で取得するようにするのがポイントな気がする。__getitem__として動作させるには。
        x = self.dataset[0][idx]
        label = self.dataset[1][idx]
 
        x = x[np.newaxis,:,:].astype(np.float32) / 255
 
        return x, label

# 学習用のデータセット
train_dataset = MyDataset([train_X, train_t])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,shuffle=True, num_workers=0)  # num_workers=0に変更
 
# 評価用のデータセット
test_dataset = MyDataset([test_X, test_t])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)  # num_workers=0に変更

best_loss = 1e5 # 誤差の初期値
n_epochs = 10 # エポック数
train_losses = []  # 追加
test_losses = []   # 追加
accuracies = []    # 追加
 
for epoch in range(1, n_epochs + 1):  # n_epochsまで含める
    print(f'Epoch = {epoch: 03d}')
 
    model.train()
    train_loss = 0
 
    # Training
    for _xs, _ts in train_loader:
        _xs, _ts = _xs.to(device), _ts.to(device)
 
        # stack overflow
        # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        opt.zero_grad() # 勾配を初期化
 
        _ys = model(_xs)
 
        loss = criterion(_ys, _ts)
        train_loss += loss.item() 
 
        # back propagation
        loss.backward() # モデルの勾配を計算
 
        opt.step() # モデルのパラメータを更新
 
        _xs = _xs.to('cpu').detach().numpy()
        _ts = _ts.to('cpu').detach().numpy()
 
    train_loss /= len(train_loader)
 
    # Evaluation
    model.eval()
 
    valid_loss = 0
    correct = 0  # 追加
    total = 0    # 追加
 
    # 評価の場合は勾配を保存しないように設定
    with torch.no_grad():
        ts, ys = [], []
        for _xs, _ts in test_loader:
            _xs, _ts = _xs.to(device), _ts.to(device)
            _ys = model(_xs)
            valid_loss += criterion(_ys,_ts).item()  # .item()追加
            
            # 精度計算のため追加
            pred = _ys.argmax(1)  # 追加
            correct += (pred == _ts).sum().item()  # 追加
            total += _ts.size(0)  # 追加
            
            ys.append(_ys.to('cpu').detach().numpy())
            ts.append(_ts.to('cpu').detach().numpy())
        ys = np.concatenate(ys, axis=0)
 
    valid_loss /= len(test_loader)
    accuracy = correct / total  # 追加
    
    # 履歴に記録  # 追加
    train_losses.append(train_loss)  # 追加
    test_losses.append(valid_loss)   # 追加
    accuracies.append(accuracy)      # 追加
 
    # 評価時の誤差関数の結果が一番小さいモデルを保存する
    if valid_loss <= best_loss:
        torch.save(model.state_dict(), 'models/model_00.pt')  # パス修正
        best_loss = valid_loss
 
    # 学習状況の確認。学習時、評価時、一番良い誤差を出力。
    print(f'Loss Train={train_loss:.4f}, Test={valid_loss:.4f}, Best={best_loss:.4f}, Acc={accuracy:.4f}')  # Acc追加

# 損失と精度のグラフを保存  # 追加
epochs_range = range(1, len(train_losses) + 1)  # 追加

plt.figure(figsize=(12, 4))  # 追加
plt.subplot(1, 2, 1)  # 追加
plt.plot(epochs_range, train_losses, label='Train Loss')  # 追加
plt.plot(epochs_range, test_losses, label='Test Loss')  # 追加
plt.title('Training and Test Loss')  # 追加
plt.legend()  # 追加
plt.xlabel('Epochs')  # 追加
plt.ylabel('Loss')  # 追加

plt.subplot(1, 2, 2)  # 追加
plt.plot(epochs_range, accuracies, label='Test Accuracy')  # 追加
plt.title('Test Accuracy')  # 追加
plt.legend()  # 追加
plt.xlabel('Epochs')  # 追加
plt.ylabel('Accuracy')  # 追加

plt.tight_layout()  # 追加
plt.savefig('results/figs/cnn_training_results.png')  # 追加
plt.close()  # 追加

# モデルのLOAD
# 公式サイト：https://pytorch.org/tutorials/beginner/saving_loading_models.html
model.load_state_dict(torch.load('models/model_00.pt'))  # パス修正
model.eval()
 
# 推論する画像の表示(確認用に表示しています。)
input_id = 0
_x = test_X[input_id,:,:].astype(np.float32) /255
 
fig, ax = plt.subplots()
ax.axes.xaxis.set_visible(False) # X軸を非表示にする
ax.axes.yaxis.set_visible(False) # Y軸を非表示にする
ax.imshow(_x, cmap='Greys')
ax.set_title(f'Test Sample (True: {test_t[input_id]})')  # タイトル追加
plt.savefig('results/figs/test_sample.png')  # 保存追加
plt.close()  # 追加
 
# 推論する画像をモデルに入力
_x = torch.from_numpy(_x[np.newaxis, np.newaxis,:,:]).clone() # 推論するためにndarray型のデータをTensor型に変換。
_x = _x.to(device) # GPUに載せる
_y = model(_x) # 推論
_y = _y.to('cpu').detach().numpy() # CPUに載せて、Tensor型からndarray型へ
 
# 推論結果を元にクラス分類
pred = np.argmax(_y, axis=1)
 
# 推論結果の表示
print(f'推論結果：{pred[0]}')  # 修正
print(f'正解ラベル：{test_t[input_id]}')  # 追加

print(f'結果が保存されました：')  # 追加
print(f'- 学習結果グラフ: results/figs/cnn_training_results.png')  # 追加
print(f'- テストサンプル: results/figs/test_sample.png')  # 追加
print(f'- 学習済みモデル: models/model_00.pt')  # 追加