import torch.nn as nn


class ClientVGG19(nn.Module):
    def __init__(self):
        super(ClientVGG19, self).__init__()
        self.client_feature_extraction = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
        )

    def forward(self, x):
        x = self.client_feature_extraction(x)
        return x

    def __str__(self):
        return "VGG19"


class ClientResNet18(nn.Module):
    def __init__(self):
        super(ClientResNet18, self).__init__()
        self.client_feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.client_feature_extraction(x)
        return x

    def __str__(self):
        return "ResNet18"


class ClientResNet34(ClientResNet18):
    def __init__(self):
        super(ClientResNet34, self).__init__()

    def __str__(self):
        return "ResNet34"
