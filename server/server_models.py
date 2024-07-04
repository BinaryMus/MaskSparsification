import torch
import torch.nn as nn


class ServerVGG19x18(nn.Module):
    def __init__(self):
        super(ServerVGG19x18, self).__init__()
        self.server_feature_extraction = nn.Sequential(
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
            nn.Sequential(
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ),
        )
        self.server_classifier = nn.Sequential(
            nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            ),
            nn.Sequential(
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5)
            ),
            nn.Sequential(
                nn.Linear(4096, 10)
            ),
        )

    def forward(self, x):
        x = self.server_feature_extraction(x)
        x = self.server_classifier(x.view(x.size(0), -1))
        return x

    def __str__(self):
        return "VGG19x1"


class ServerVGG19X17(ServerVGG19x18):
    def __init__(self):
        super(ServerVGG19X17, self).__init__()
        del self.server_feature_extraction[0]

    def __str__(self):
        return "VGG19x2"


class ServerVGG19X11(ServerVGG19x18):
    def __init__(self):
        super(ServerVGG19X11, self).__init__()
        del self.server_feature_extraction[:7]

    def __str__(self):
        return "VGG19x8"


class ServerVGG19x4(ServerVGG19x18):
    def __init__(self):
        super(ServerVGG19x4, self).__init__()
        del self.server_feature_extraction[:14]

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
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ServerResNet18x17(nn.Module):
    def __init__(self):
        super(ServerResNet18x17, self).__init__()

        self.in_channels = 64
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 100)

        self.server_feature_extraction = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avg_pool,
        )
        self.server_classifier = nn.Linear(512, 100)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.server_feature_extraction(x)
        x = self.server_classifier(x.view(x.size(0), -1))
        return x

    def __str__(self):
        return "ResNet18x1"


class ServerResNet18x16(ServerResNet18x17):
    def __init__(self):
        super(ServerResNet18x16, self).__init__()
        del self.layer1[0]

    def __str__(self):
        return "ResNet18x2"


class ServerResNet18x9(ServerResNet18x17):
    def __init__(self):
        super(ServerResNet18x9, self).__init__()
        del self.layer1
        del self.layer2
        self.server_feature_extraction = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avg_pool,
        )

    def __str__(self):
        return "ResNet18x9"


class ServerResNet18x5(ServerResNet18x17):
    def __init__(self):
        super(ServerResNet18x5, self).__init__()
        del self.layer1
        del self.layer2
        self.server_feature_extraction = nn.Sequential(
            self.layer4,
            self.avg_pool,
        )

    def __str__(self):
        return "ResNet18x13"


class ServerResNet34x33(ServerResNet18x17):
    def __init__(self):
        super(ServerResNet34x33, self).__init__()
        self.in_channels = 64
        self.layer1 = self._make_layer(BasicBlock, 64, 3, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, 2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, 2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.server_feature_extraction = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.avgpool,
        )
        self.server_classifier = nn.Linear(512, 1000)

    def __str__(self):
        return "ResNet34x1"


class ServerResNet34x32(ServerResNet34x33):
    def __init__(self):
        super(ServerResNet34x32, self).__init__()
        del self.layer1[0]

    def __str__(self):
        return "ResNet34x2"


class ServerResNet34x19(ServerResNet34x33):
    def __init__(self):
        super(ServerResNet34x19, self).__init__()
        del self.layer1
        del self.layer2
        self.server_feature_extraction = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        )

    def __str__(self):
        return "ResNet34x15"


class ServerResNet34x7(ServerResNet34x33):
    def __init__(self):
        super(ServerResNet34x7, self).__init__()
        del self.layer1
        del self.layer2
        del self.layer3
        self.server_feature_extraction = nn.Sequential(
            self.layer4,
            self.avgpool,
        )

    def __str__(self):
        return "ResNet34x27"
