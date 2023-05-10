import torch
import torch.nn as nn


class ClientVGG19x1(nn.Module):
    def __init__(self):
        super(ClientVGG19x1, self).__init__()
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
        return "VGG19x1"


class ClientVGG19x2(ClientVGG19x1):
    def __init__(self):
        super(ClientVGG19x2, self).__init__()
        self.client_feature_extraction.append(
            nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ),
        )

    def __str__(self):
        return "VGG19x2"


class ClientVGG19x8(ClientVGG19x1):
    def __init__(self):
        super(ClientVGG19x8, self).__init__()
        self.client_feature_extraction.append(
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
            )
        )

    def __str__(self):
        return "VGG19x8"


class ClientVGG19x15(ClientVGG19x1):
    def __init__(self):
        super(ClientVGG19x15, self).__init__()
        self.client_feature_extraction.append(
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True),
                ),
            )
        )

    def __str__(self):
        return "VGG19x15"


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        import torch.nn.functional as F
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ClientResNet18x1(nn.Module):
    def __init__(self):
        super(ClientResNet18x1, self).__init__()
        self.client_feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.client_feature_extraction(x)
        return x

    def __str__(self):
        return "ResNet18x1"


class ClientResNet18x2(ClientResNet18x1):
    def __init__(self):
        super(ClientResNet18x2, self).__init__()
        self.client_feature_extraction.append(
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )
        )

    def __str__(self):
        return "ResNet18x2"


class ClientResNet18x9(ClientResNet18x1):
    def __init__(self):
        super(ClientResNet18x9, self).__init__()
        self.in_channels = 64
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)
        self.client_feature_extraction.append(
            nn.Sequential(
                self.layer1,
                self.layer2,
            )
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def __str__(self):
        return "ResNet18x9"


class ClientResNet18x13(ClientResNet18x9):
    def __init__(self):
        super(ClientResNet18x13, self).__init__()
        self.layer3 = self._make_layer(BasicBlock, 256, 2, 2)
        self.client_feature_extraction.append(self.layer3)

    def __str__(self):
        return "ResNet18x13"


class ClientResNet34x1(nn.Module):
    def __init__(self):
        super(ClientResNet34x1, self).__init__()
        self.client_feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.client_feature_extraction(x)
        return x

    def __str__(self):
        return "ResNet34x1"


class ClientResNet34x2(ClientResNet34x1):
    def __init__(self):
        super(ClientResNet34x2, self).__init__()
        self.client_feature_extraction.append(
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU()
            )
        )

    def __str__(self):
        return "ResNet34x2"


class ClientResNet34x15(ClientResNet34x1):
    def __init__(self):
        super(ClientResNet34x15, self).__init__()
        self.in_channels = 64
        self.client_feature_extraction.append(
            nn.Sequential(
                self._make_layer(BasicBlock, 64, 2, 1),
                self._make_layer(BasicBlock, 128, 2, 2),
            )
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def __str__(self):
        return "ResNet34x15"


class ClientResNet34x27(ClientResNet34x15):
    def __init__(self):
        super(ClientResNet34x27, self).__init__()
        self.client_feature_extraction.append(
            nn.Sequential(
                self._make_layer(BasicBlock, 256, 6, 2),
            )
        )

    def __str__(self):
        return "ResNet34x27"
