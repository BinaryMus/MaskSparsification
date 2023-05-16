import argparse
import os
import random

import numpy as np
import torch


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
    parser.add_argument('--role', default='client')  # client / server
    parser.add_argument('--arch', default='vgg19_cifar10')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--ip', default='127.0.0.1')
    parser.add_argument('--port', default=8000)
    parser.add_argument('--server_ip', default='127.0.0.1')
    parser.add_argument('--server_port', default=9000)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--compressor', default='baseline')
    parser.add_argument('--bit', default=8, type=int)
    parser.add_argument('--ratio', default=0.9, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--path', default='../datasets')
    parser.add_argument('--cutlayer', default=1, type=int)
    arg = parser.parse_args()
    if arg.role == 'client':
        from client import *

        arch = {"vgg19_cifar10": vgg19_cifar10_client,
                "resnet18_cifar100": resnet18_cifar100_client,
                "resnet34_tiny_imagenet200": resnet34_tiny_imagenet200_client}

        encode = {"sparsification": EncodeSparser,
                  "quantization": EncodeQuantizer,
                  "masked_sparsification": EncodeMaskedSparser}

        compressor = encode.get(arg.compressor)
        if compressor is not None:
            compressor = compressor(bit=arg.bit, ratio=arg.ratio)
        client = arch.get(arg.arch)(
            device=arg.device,
            batch_size=arg.batch_size,
            path=arg.path,
            ip=arg.ip,
            port=arg.port,
            server_ip=arg.server_ip,
            server_port=arg.server_port,
            epoch=arg.epoch,
            cutlayer=arg.cutlayer,
            compressor=compressor,
        )
        print(arg.arch)
        print(arg.compressor)
        client.train()
    elif arg.role == 'server':
        from server import *

        arch = {"vgg19_cifar10": vgg19_cifar10_server,
                "resnet18_cifar100": resnet18_cifar100_server,
                "resnet34_tiny_imagenet200": resnet34_tiny_imagenet200_server}

        decode = {"sparsification": DecodeSparser,
                  "quantization": DecodeQuantizer,
                  "masked_sparsification": DecodeMaskedSparser}
        compressor = decode.get(arg.compressor)
        if compressor is not None:
            compressor = compressor(bit=arg.bit, ratio=arg.ratio)
        server = arch.get(arg.arch)(
            device=arg.device,
            server_ip=arg.server_ip,
            server_port=arg.server_port,
            batch_size=arg.batch_size,
            compressor=compressor,
            cutlayer=arg.cutlayer,
            epoch=arg.epoch,
        )
        print(arg.arch)
        print(arg.compressor)
        server.train()
