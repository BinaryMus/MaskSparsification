import argparse
import os
import random

import numpy as np
import torch

from functools import reduce

def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


if __name__ == '__main__':
    seed_it(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', default='client')    
    parser.add_argument('--arch', default='vgg19_cifar10')
    parser.add_argument('--cutlayer', type=str)
    parser.add_argument('--path', type=str)

    parser.add_argument('--compressor', type=str)
    parser.add_argument('--ratio', type=float)
    parser.add_argument('--bit', type=int)

    parser.add_argument('--client_ip', default='127.0.0.1')
    parser.add_argument('--client_port', default=8000)
    parser.add_argument('--client_device', default='cpu')
    parser.add_argument('--server_ip', default='127.0.0.1')
    parser.add_argument('--server_port', default=9000)
    parser.add_argument('--server_device', type=str, default='cuda')

    parser.add_argument('--batch_size', default=256, type=int)
    arg = parser.parse_args()

    sizes_vgg19 = {'shallow': [arg.batch_size, 64, 16, 16], 'medium': [arg.batch_size, 256, 4, 4], 'deep': [arg.batch_size, 512, 2, 2]}
    sizes_res18 = {'shallow': [arg.batch_size, 64, 32, 32], 'medium': [arg.batch_size, 128, 16, 16], 'deep': [arg.batch_size, 256, 8, 8]}
    sizes_res34 = {'shallow': [arg.batch_size, 64, 56, 56], 'medium': [arg.batch_size, 128, 28, 28], 'deep': [arg.batch_size, 256, 14, 14]}

    sizes = {"vgg19_cifar10": sizes_vgg19, "res18_cifar100": sizes_res18, "res34_imagenet": sizes_res34}

    if arg.role == 'client':
        from client.client import *
        from client.client_encode import *

        arch = {"vgg19_cifar10": vgg19_cifar10_client,
                "res18_cifar100": resnet18_cifar100_client,
                "res34_imagenet": resnet34_imagenet_client}

        encode = {"sp": EncodeSparser,
                  "qu": EncodeQuantizer,
                  "ms": EncodeMaskedSparser}
        compressor = encode.get(arg.compressor)
        if compressor is not None:
            k = int(reduce((lambda x, y: x * y), sizes[arg.arch][arg.cutlayer]) * (1 - arg.ratio))
            compressor = compressor(bit=arg.bit, k=k)
        client = arch.get(arg.arch)(
            device=arg.client_device,
            batch_size=arg.batch_size,
            path=arg.path,
            ip=arg.client_ip,
            port=arg.client_port,
            server_ip=arg.server_ip,
            server_port=arg.server_port,
            cutlayer=arg.cutlayer,
            compressor=compressor,
        )
        client.train()
    elif arg.role == 'server':
        from server.server import *
        from server.server_decode import *

        arch = {"vgg19_cifar10": vgg19_cifar10_server,
                "res18_cifar100": resnet18_cifar100_server,
                "res34_imagenet": resnet34_imagenet_server}

        decode = {"sp": DecodeSparser,
                  "qu": DecodeQuantizer,
                  "ms": DecodeMaskedSparser}
        compressor = decode.get(arg.compressor)
        if compressor is not None:
            k = int(reduce((lambda x, y: x * y), sizes[arg.arch][arg.cutlayer]) * (1 - arg.ratio))
            compressor = compressor(bit=arg.bit, k=k)
        server = arch.get(arg.arch)(
            device=arg.server_device,
            server_ip=arg.server_ip,
            server_port=arg.server_port,
            batch_size=arg.batch_size,
            compressor=compressor,
            cutlayer=arg.cutlayer,
        )
        server.train()
