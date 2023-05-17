# Sparse but Accurate: Communication-Efficient Collaborative Learning via Sparsified Features with Mask Encoding

## Parameter Description

**role**:('client' or 'server') String type, specifying whether the current device is a server or a client \
**arch**:('vgg19_cifar10' or 'resnet18_cifar100' or 'resnet34_tiny_imagenet200') String type, specifying the data set for model training \
**device**:('cuda' or 'cpu') String type, specifying the equipment to be used \
**ip**: String type, specifying the client's ip [*only the client needs to specify*] \
**port**: Int type, specifying the client's port [*only the client needs to specify*] \
**server_ip**: String type, specifying the server's ip \
**server_port**: Int type, specifying the server's port \
**batch_size**: Int type: specifying the mini-batch size \
**compressor**:('sparsification' or 'quantization' or 'mask_sparsification') String type, specifying compression policy \
**bit**: Int type, specifying quantization's bit or sparse compensation mask's bit \
**ratio**: Float type, specifying sparsification's ratio \
**epoch**: Int type specifying training rounds \
**path**: String type, specifying the dataset path [*only the client needs to specify*] \
**cutlayer**: Int type, specifying  the cutlayer position




## Examples

### VGG19 on CIFAR-10

```
# Baseline

python3 main.py --role='server' --arch='vgg19_cifar10' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='baseline' --epoch=40 --cutlayer=1 # Server
python3 main.py --role='client' --arch='vgg19_cifar10' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='baseline' --epoch=40 --path='../datasets' --cutlayer=1 # Client

# 2-bit quantization

python3 main.py --role='server' --arch='vgg19_cifar10' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='quantization' --bit=2 --epoch=40 --cutlayer=1 # Server
python3 main.py --role='client' --arch='vgg19_cifar10' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='quantization' --bit=2 --epoch=40 --path='../datasets' --cutlayer=1 # Client

# 96.4% sparsification

python main.py --role='server' --arch='vgg19_cifar10' --device='cuda' --server_ip='127.0.0.1' --server_port=9001 --batch_size=256 --compressor='sparsification' --ratio=0.964 --epoch=40 --cutlayer=1 # Server
python main.py --role='client' --arch='vgg19_cifar10' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9001 --batch_size=256 --compressor='sparsification' --ratio=0.964 --epoch=40 --path='../datasets' --cutlayer=1 # Client

# 99% sparsification with 2-bit compensation

python3 main.py --role='server' --arch='vgg19_cifar10' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification_compensation' --ratio=0.99 --bit=2 --epoch=40 --cutlayer=1 # Server
python3 main.py --role='client' --arch='vgg19_cifar10' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification_compensation' --ratio=0.99 --bit=2 --epoch=40 --path='../datasets' --cutlayer=1 # Client

```

### ResNet18 on CIFAR-100

```
# Baseline

python3 main.py --role='server' --arch='resnet18_cifar100' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='baseline' --epoch=60 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet18_cifar100' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='baseline' --epoch=60 --path='../datasets' --cutlayer=1 # Client

# 2-bit quantization

python3 main.py --role='server' --arch='resnet18_cifar100' --device='cpu' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='quantization' --bit=2 --epoch=60 --cutlayer=1# Server
python3 main.py --role='client' --arch='resnet18_cifar100' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='quantization' --bit=2 --epoch=60 --path='../datasets' --cutlayer=1 # Client

# 96.4% sparsification

python3 main.py --role='server' --arch='resnet18_cifar100' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification' --ratio=0.964 --epoch=60 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet18_cifar100' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification' --ratio=0.964 --epoch=60 --path='../datasets' --cutlayer=1 # Client

# 99% sparsification with 2-bit compensation

python3 main.py --role='server' --arch='resnet18_cifar100' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification_compensation' --ratio=0.99 --bit=2 --epoch=60 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet18_cifar100' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification_compensation' --ratio=0.99 --bit=2 --epoch=60 --path='../datasets' --cutlayer=1 # Client

```

### ResNet34 on Tiny-ImageNet

```
# Baseline

python3 main.py --role='server' --arch='resnet34_tiny_imagenet200' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='baseline' --epoch=90 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet34_tiny_imagenet200' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='baseline' --epoch=90 --path='../datasets' --cutlayer=1 # Client

# 2-bit quantization

python3 main.py --role='server' --arch='resnet34_tiny_imagenet200' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='quantization' --bit=2 --epoch=90 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet34_tiny_imagenet200' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='quantization' --bit=2 --epoch=90 --path='../datasets' --cutlayer=1 # Client

# 96.4% sparsification

python3 main.py --role='server' --arch='resnet34_tiny_imagenet200' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification' --ratio=0.964 --epoch=90 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet34_tiny_imagenet200' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification' --ratio=0.964 --epoch=90 --path='../datasets' --cutlayer=1 # Client

# 99% sparsification with 2-bit compensation

python3 main.py --role='server' --arch='resnet34_tiny_imagenet200' --device='cuda' --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification_compensation' --ratio=0.99 --bit=2 --epoch=90 --cutlayer=1 # Server
python3 main.py --role='client' --arch='resnet34_tiny_imagenet200' --device='cpu' --ip='127.0.0.1' --port=8000 --server_ip='127.0.0.1' --server_port=9000 --batch_size=256 --compressor='sparsification_compensation' --ratio=0.99 --bit=2 --epoch=90 --path='../datasets' --cutlayer=1 # Client

```