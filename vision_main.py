import argparse
import torch
import numpy as np

from utils import seed_it
from utils import vision_train as train
from utils import vision_validate as validate
from data import cifar10, cifar100
from models import vgg16, resnet18
from compression import Baseline, VanillaQuantization, QuantileQuantization, FpQuantization, VanillaSparsification, MaskSparsification, RandTopkSparsification

seed_it(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['vgg16', 'resnet18'])
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'])
parser.add_argument('--num_class', type=int)
parser.add_argument('--cutlayer', type=int)
parser.add_argument('--comp', type=str, choices=['baseline', 'vq', 'qq', 'fpq', 'vs', 'ms', 'rts'])

parser.add_argument('--device', type=str, default='cuda')

parser.add_argument('--r', type=float, default=0.01)
parser.add_argument('--b', type=int, default=2)
parser.add_argument('--offset', type=float, default=0.99)
parser.add_argument('--ebit', type=int, default=1)
parser.add_argument('--mbit', type=int, default=1)
parser.add_argument('--positive', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.1)

arg = parser.parse_args()

client, server, model, optimizer_c, optimizer_s, scheduler_c, scheduler_s \
    = {
        'vgg16': vgg16, 
        'resnet18': resnet18, 
    }[arg.model](arg.cutlayer, arg.num_class)

client = client.to(arg.device)
server = server.to(arg.device)

train_loader, validate_loader \
    = {
        'cifar10': cifar10,
        'cifar100': cifar100,
    }[arg.dataset]()

comp \
    = {
        'baseline': Baseline,
        'vq': VanillaQuantization,
        'qq': QuantileQuantization,
        'fpq': FpQuantization,
        'vs': VanillaSparsification,
        'ms': MaskSparsification,
        'rts': RandTopkSparsification
    }[arg.comp](ratio=arg.r, bit=arg.b, offset=arg.offset, exponent_bits=arg.ebit, mantissa_bits=arg.mbit, positive=arg.positive, alpha=arg.alpha)

criterion = torch.nn.CrossEntropyLoss()

loss_lst = []
acc1_lst = []
acc5_lst = []

for i in range(200):
    loss = train(i, client, optimizer_c, server, optimizer_s, train_loader, arg.device, comp, criterion)
    scheduler_c.step()
    scheduler_s.step()
    acc1, acc5 = validate(client, server, validate_loader, arg.device, comp)
    loss_lst.append(loss)
    acc1_lst.append(acc1)
    acc5_lst.append(acc5)

np.save(f'./result/{arg.model}_{arg.cutlayer}_{arg.dataset}_{arg.comp}_loss.npy', np.array(loss_lst))
np.save(f'./result/{arg.model}_{arg.cutlayer}_{arg.dataset}_{arg.comp}_acc1.npy', np.array(acc1_lst))
np.save(f'./result/{arg.model}_{arg.cutlayer}_{arg.dataset}_{arg.comp}_acc5.npy', np.array(acc5_lst))
