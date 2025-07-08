import argparse
import torch
import numpy as np

from utils import seed_it
from utils import nlp_train as train
from utils import nlp_validate as validate

from data import wikitext2
from models import llama3, qwen3
from compression import Baseline, VanillaQuantization, QuantileQuantization, FpQuantization, VanillaSparsification, MaskSparsification, RandTopkSparsification

seed_it(42)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['llama', 'qwen'])
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

client, server, model, tokenizer, optimizer_c, optimizer_s \
    = {
        'llama': llama3, 
        'qwen': qwen3
    }[arg.model](arg.cutlayer, arg.device)

client = client.to(arg.device)
server = server.to(arg.device)

vocab_size = model.config.vocab_size

train_loader, validate_loader = wikitext2(tokenizer, bs=4)

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
ppl_lst = []

_, ppl = validate(client, server, validate_loader, arg.device, comp, criterion, vocab_size)
ppl_lst.append(ppl)

for i in range(3):
    loss = train(i, client, optimizer_c, server, optimizer_s, train_loader, arg.device, comp, criterion, vocab_size)
    _, ppl = validate(client, server, validate_loader, arg.device, comp, criterion, vocab_size)
    loss_lst.append(loss)
    ppl_lst.append(ppl)

np.save(f'./result/{arg.model}_{arg.cutlayer}_{arg.comp}_loss.npy', np.array(loss_lst))
np.save(f'./result/{arg.model}_{arg.cutlayer}_{arg.comp}_ppl.npy', np.array(ppl_lst))
