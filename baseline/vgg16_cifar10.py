import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR10
import sys
sys.path.append('..')
from models import vgg16

device = 'cuda:0'
c, s, model, optimizer_c, optimizer_s, scheduler_c, scheduler_s = vgg16(1, 10)
model = model.to(device)

transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = Compose([
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
train_set = CIFAR10(root='/dataset/cifar10', train=True, transform=transform_train, download=True)
validate_set = CIFAR10(root='/dataset/cifar10', train=False, transform=transform_test)
train_loader = DataLoader(train_set, 256, shuffle=True)
validate_loader = DataLoader(validate_set, 256, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = torch.nn.CrossEntropyLoss()

loss_lst = []
acc1_lst = []
acc5_lst = []

def train(epoch):
    model.train()
    running_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / total_batches:.0f}%)]\tLoss: {loss.item():.6f}')

    average_loss = running_loss / total_batches
    print(f'Epoch {epoch} Average Loss: {average_loss:.4f}')
    loss_lst.append(average_loss)

def validate():
    model.eval()
    total, correct1, correct5 = 0, 0, 0
    for data, target in validate_loader:
        data = data.to(device)
        target = target.to(device)
        total += len(data)
        output = model(data)
        predict = output.argmax(dim=1)
        correct1 += torch.eq(predict, target).sum().float().item()
        target_resize = target.view(-1, 1)
        _, predict = output.topk(5)
        correct5 += torch.eq(predict, target_resize).sum().float().item()
    acc1 = correct1 / total
    acc5 = correct5 / total
    print(f'\nTop-1 Accuracy: {acc1}, Top-5 Accuracy: {acc5}\n')
    acc1_lst.append(acc1)
    acc5_lst.append(acc5)

for i in range(200):
    train(i)
    validate()
    scheduler.step()

np.save(f'./vgg16_cifar10_baseline_loss.npy', np.array(loss_lst))
np.save(f'./vgg16_cifar10_baseline_acc1.npy', np.array(acc1_lst))
np.save(f'./vgg16_cifar10_baseline_acc5.npy', np.array(acc5_lst))