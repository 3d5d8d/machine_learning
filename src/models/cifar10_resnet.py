from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn

def create_cifar10_resnet18(pretrained = True): #imagenet1Kで学習済みの物を使うか否かを選択
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights = weights)
    model.conv1 = nn.Conv2d(
        in_channels = 3,
        out_channels = 64,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        bias=False, #convのあとにbatchnormの処理してるので、biasをここでいれてもどうせずらされる.
    )
    model.maxpool=nn.Identity()
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model

#weights = ResNet18_Weights.DEFAULT
#model = resnet18(weights = weights)
#print(model)