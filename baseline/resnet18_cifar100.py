import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
from torchvision.datasets import CIFAR100
import sys
sys.path.append('..')
from models import resnet18

device = 'cuda:1'
c, s, model, optimizer_c, optimizer_s, scheduler_c, scheduler_s = resnet18(1, 100)
model = model.to(device)

transform_train = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
transform_test = Compose([
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
train_set = CIFAR100(root='/dataset/cifar100', train=True, transform=transform_train, download=True)
validate_set = CIFAR100(root='/dataset/cifar100', train=False, transform=transform_test)
train_loader = DataLoader(train_set, 256, shuffle=True)
validate_loader = DataLoader(validate_set, 256, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
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
        output = s(c(data))
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
    # scheduler.step()
    scheduler_c.step()
    scheduler_s.step()

np.save(f'./resnet18_cifar100_baseline_loss.npy', np.array(loss_lst))
np.save(f'./resnet18_cifar100_baseline_acc1.npy', np.array(acc1_lst))
np.save(f'./resnet18_cifar100_baseline_acc5.npy', np.array(acc5_lst))