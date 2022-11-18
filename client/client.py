import pickle

import numpy as np
import torch
import zmq
from torch.utils.data import DataLoader


class BaseClient:
    def __init__(self,
                 epoch: int,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 train_loader: DataLoader,
                 validate_loader: DataLoader,
                 ip: str = "127.0.0.1",
                 port: int = 8000,
                 server_ip: str = "127.0.0.1",
                 server_port: int = 9000,
                 device: str = "cpu",
                 compressor=None,
                 ):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.validate_loader = validate_loader
        self.device = torch.device(device)
        self.compressor = compressor

        self.ip = ip
        self.port = port
        self.server_ip = server_ip
        self.server_port = server_port
        self.context = zmq.Context()

        self.acc1 = []
        self.acc5 = []
        print(f"{str(self)} start!!!")

    def run(self, x, y, mode="train"):
        x = x.to(self.device)

        client_socket = self.context.socket(zmq.REQ)
        client_socket.connect("tcp://" + self.server_ip + ":" + str(self.server_port))

        if mode == "train":
            self.model.train()
            self.optimizer.zero_grad()
            smashed_data = self.model(x)
        else:
            self.model.eval()
            with torch.no_grad():
                smashed_data = self.model(x)

        if self.compressor is not None:
            v, m = self.compressor.encode(smashed_data)
            data = pickle.dumps((mode, v, m, y))
            client_socket.send(data)
        else:
            data = pickle.dumps((mode, smashed_data.data, y))
            client_socket.send(data)
        data = pickle.loads(client_socket.recv()).to(self.device)
        if mode == "train":
            smashed_data.backward(data)
            self.optimizer.step()
        client_socket.close()
        if mode == "eval":
            return data

    def train(self):
        for i in range(self.epoch):
            for x, y in self.train_loader:
                self.run(x, y, "train")
            self.scheduler.step()
            acc1, acc5 = self.validate()
            self.acc1.append(acc1)
            self.acc5.append(acc5)
            print(
                f"{str(self)} "
                f"Epoch[{i + 1}|{self.epoch}] "
                f"Top-1 Accuracy {acc1} "
                f"Top-5 Accuracy {acc5}"
            )
        np.save(f'./res/{str(self.model)}_{str(self.compressor)}_acc1.npy',
                np.array(self.acc1))
        np.save(f'./res/{str(self.model)}_{str(self.compressor)}_acc5.npy',
                np.array(self.acc5))

    def validate(self):
        total, correct1, correct5 = 0, 0, 0
        for x, y in self.validate_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            total += len(x)
            output = self.run(x, y, "eval").to(self.device)
            predict = output.argmax(dim=1)
            correct1 += torch.eq(predict, y).sum().float().item()
            y_resize = y.view(-1, 1)
            _, predict = output.topk(5)
            correct5 += torch.eq(predict, y_resize).sum().float().item()
        return correct1 / total, correct5 / total

    def __str__(self):
        return f"Client@{self.ip}:{self.port}"


def vgg19_cifar10_client(device: str = 'cpu',
                         batch_size: int = 256,
                         path: str = '../datasets',
                         ip: str = "127.0.0.1",
                         port: int = 8000,
                         server_ip: str = "127.0.0.1",
                         server_port: int = 9000,
                         epoch: int = 40,
                         compressor=None,
                         ):
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    from .client_models import ClientVGG19

    model = ClientVGG19().to(torch.device(device))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_set = CIFAR10(root=path, train=True, transform=transform)
    validate_set = CIFAR10(root=path, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size, shuffle=False)
    return BaseClient(epoch, model, optimizer, scheduler, train_loader, validate_loader, ip, port, server_ip,
                      server_port,
                      device, compressor)


def resnet18_cifar100_client(device: str = 'cpu',
                             batch_size: int = 256,
                             path: str = '../datasets',
                             ip: str = "127.0.0.1",
                             port: int = 8000,
                             server_ip: str = "127.0.0.1",
                             server_port: int = 9000,
                             epoch: int = 60,
                             compressor=None,
                             ):
    from torchvision.datasets import CIFAR100
    from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation
    from .client_models import ClientResNet18
    model = ClientResNet18().to(torch.device(device))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)

    transform = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ToTensor(),
        Normalize((0.5070, 0.4865, 0.4409)
                  , (0.2673, 0.2564, 0.2761))]
    )

    train_set = CIFAR100(root=path, train=True, transform=transform)
    validate_set = CIFAR100(root=path, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size, shuffle=False)

    return BaseClient(epoch, model, optimizer, scheduler, train_loader, validate_loader, ip, port, server_ip,
                      server_port,
                      device, compressor)


def resnet34_tiny_imagenet200_client(device: str = 'cpu',
                                     batch_size: int = 256,
                                     path: str = '/dataset/tiny-imagenet-200',
                                     ip: str = "127.0.0.1",
                                     port: int = 8000,
                                     server_ip: str = "127.0.0.1",
                                     server_port: int = 9000,
                                     epoch: int = 90,
                                     compressor=None,
                                     ):
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, RandomRotation
    transform = Compose([
        RandomCrop(64, padding=4),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    train_set = ImageFolder(
        root=path + '/train',
        transform=transform)
    validate_set = ImageFolder(
        root=path + '/val',
        transform=transform)

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    validate_loader = DataLoader(validate_set, batch_size, shuffle=False)

    from .client_models import ClientResNet34
    model = ClientResNet34().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    return BaseClient(epoch, model, optimizer, scheduler, train_loader, validate_loader, ip, port, server_ip,
                      server_port,
                      device, compressor)
