import pickle

import numpy as np
import torch
import zmq


class BaseServer:
    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler,
                 sz,
                 epoch: int,
                 server_ip: str = '127.0.0.1',
                 server_port: int = 9000,
                 device: str = 'cuda',
                 compressor=None):
        self.epoch = epoch
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device(device)
        self.compressor = compressor
        self.sz = sz

        self.cur = 1
        self.cnt = 0
        self.loss = 0
        self.last = "train"
        self.loss_lst = []

        self.server_ip = server_ip
        self.server_port = server_port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://" + self.server_ip + ":" + str(self.server_port))

        print(f"{str(self)} start!!!")

    def run(self):
        if self.compressor is not None:
            mode, v, m, y = pickle.loads(self.socket.recv())
            self.sz[0] = len(y)
            smashed_data = torch.zeros(self.sz).to(self.device)
            self.compressor(smashed_data, v, m)
            smashed_data = smashed_data.to(self.device)
        else:
            mode, smashed_data, y = pickle.loads(self.socket.recv())
            smashed_data = smashed_data.to(self.device)
        y = y.to(self.device)

        if mode == "train":
            self.cnt += 1
            self.model.train()
            smashed_data.requires_grad = True
            self.optimizer.zero_grad()
            output = self.model(smashed_data)
            loss = self.criterion(output, y)
            loss.backward()
            self.loss += loss.item()
            if self.cnt % 50 == 0:
                print(f"{str(self)} Index: {self.cnt} Loss: {round(self.loss / self.cnt, 3)}")
            self.optimizer.step()
            data = smashed_data.grad
        else:
            if self.last == "train":
                self.loss_lst.append(self.loss / self.cnt)
                print(f"{str(self)} Epoch[{self.cur}|{self.epoch}] Average loss: {round(self.loss_lst[-1], 3)}")
                self.scheduler.step()
                self.loss = 0
                self.cnt = 0
                self.cur += 1
            with torch.no_grad():
                self.model.eval()
                data = self.model(smashed_data)
        self.last = mode
        self.socket.send(pickle.dumps(data))

    def train(self):
        flag = True
        while True:
            self.run()
            if len(self.loss_lst) == self.epoch and flag:
                np.save(f'./res/{str(self.model)}_{str(self.compressor)}_loss.npy', np.array(self.loss_lst))
                flag = False

    def __str__(self):
        return f"Server@{self.server_ip}:{self.server_port}"


def vgg19_cifar10_server(device: str = 'cuda',
                         server_ip: str = "127.0.0.1",
                         server_port: int = 9000,
                         batch_size: int = 256,
                         compressor=None,
                         cutlayer: str='shallow',
                         ):
    from .server_models import  ServerVGG19X17, ServerVGG19X11, ServerVGG19x4
    models = {'shallow': ServerVGG19X17, 'medium': ServerVGG19X11, 'deep': ServerVGG19x4}
    model = models[cutlayer]().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    sizes = {'shallow': [batch_size, 64, 16, 16], 'medium': [batch_size, 256, 4, 4], 'deep': [batch_size, 512, 2, 2]}

    return BaseServer(model, optimizer, scheduler, sizes[cutlayer], 200, server_ip, server_port, device, compressor)


def resnet18_cifar100_server(device: str = 'cuda',
                             server_ip: str = "127.0.0.1",
                             server_port: int = 9000,
                             batch_size: int = 256,
                             compressor=None,
                             cutlayer: str='shallow',
                             ):
    from .server_models import ServerResNet18x16, ServerResNet18x9, ServerResNet18x5
    models = {'shallow': ServerResNet18x16, 'medium': ServerResNet18x9, 'deep': ServerResNet18x5}
    model = models[cutlayer]().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    sizes = {'shallow': [batch_size, 64, 32, 32], 'medium': [batch_size, 128, 16, 16], 'deep': [batch_size, 256, 8, 8]}
    return BaseServer(model, optimizer, scheduler, sizes[cutlayer], 200, server_ip, server_port, device, compressor)
                      
def resnet34_imagenet_server(device: str = 'cuda',
                             server_ip: str = "127.0.0.1",
                             server_port: int = 9000,
                             batch_size: int = 256,
                             compressor=None,
                             cutlayer: str = 'shallow',
                             ):
    from .server_models import ServerResNet34x32, ServerResNet34x19, ServerResNet34x7
    models = {'shallow': ServerResNet34x32, 'medium': ServerResNet34x19, 'deep': ServerResNet34x7}
    model = models[cutlayer]().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    sizes = {'shallow': [batch_size, 64, 56, 56], 'medium': [batch_size, 128, 28, 28], 'deep': [batch_size, 256, 14, 14]}
    return BaseServer(model, optimizer, scheduler, sizes[cutlayer], 90, server_ip, server_port, device, compressor)