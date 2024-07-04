# Mask-Encoded Sparsification: Mitigating Biased Gradients in Communication-Efficient Split Learning

This paper introduces a novel framework designed to achieve a high compression ratio in Split Learning (SL) scenarios where resource-constrained devices are involved in large-scale model training. Our investigations demonstrate that compressing feature maps within SL leads to biased gradients that can negatively impact the convergence rates and diminish the generalization capabilities of the resulting models. Our theoretical analysis provides insights into how compression errors critically hinder SL performance, which previous methodologies underestimate. To address these challenges, we employ a narrow bit-width encoded mask to compensate for the sparsification error without increasing the order of time complexity. Supported by rigorous theoretical analysis, our framework significantly reduces compression errors and accelerates the convergence. Extensive experiments also verify that our method outperforms existing solutions regarding training efficiency and communication complexity.

🎉 **Our paper has been accepted at the 27th European Conference on Artifical Intelligence (ECAI 2024)** 🎉

Manuscript Paper: [paper](https://github.com/0xc0de996/MaskSparsification/blob/master/paper_manuscript.pdf)

# How to Use
Install packages in `requirements.txt`:
```shell
pip install -r requirements.txt
```

## Parameter Description in `main.py`
- role: Specify role as **client** or **server**, prevent client and server models separately
- arch: Specify the model and dataset, optional architectures include:
    - vgg19_cifar10: Using vgg19 to train the CIFAR10 dataset
    - res18_cifar100: Using resnet18 to train the CIFAR100 dataset
    - res34_imagenet: Using resnet34 to train the ImageNet1K dataset
- cutlayer: The cut position of neural network, optional cutlayer include:
    - shallow: Vgg19, resnet18, and resnet34 are cut on the 2nd, 2nd, and 2nd layers respectively
    - medium: Vgg19, resnet18, and resnet34 are cut on the 8th, 9th, and 15th layers respectively
    - deep:  Vgg19, resnet18, and resnet34 are cut on the 15th, 13th, and 27th layers respectively
- path: The dataset path
- compressor: Compress feature maps to reduce communication volume, optional methods include:
    - qu: Quantization compression
    - sp: Top-k sparsification compression
    - ms: **Mask sparsification (Ours)**
    - others: No compression
- ratio: Sparsification ratio, effective for sp and ms, k will automatically calculate based on the size of the feature map
- bit: Width of quantization bit or mask in ms, effective for qu and ms 
- client_ip: The IP address of client，which can connect to the server 
- client_port: The port of client
- client_device: The device of client, optional devices include:
    - cpu
    - cuda
- server_ip: The IP address of server, which can connect to the client
- server_port: The port of server
- server_device: The device of server, optional devices include:
    - cpu
    - cuda
- batch_size: The batch_size during the training progress

## Example: Using VGG19 to train the CIFAR10 dataset

### Local multi process execution
Open a termianal:

```shell
# Progress 1: Server
python main.py --role server --arch vgg19_cifar10 --cutlayer deep \
--compressor ms --ratio 0.99 --bit 2 \
--client_ip 127.0.0.1 --client_port 8000 --client_device cpu \
--server_ip 127.0.0.1 --server_port 9000 --server_device cuda \
--batch_size 256
```

Open a new terminal:

```shell
# Progress 2: Client
python main.py --role client --arch vgg19_cifar10 --cutlayer deep --path your_dataser_path \
--compressor ms --ratio 0.99 --bit 2 \
--client_ip 127.0.0.1 --client_port 8000 --client_device cpu \
--server_ip 127.0.0.1 --server_port 9000 --server_device cuda \
--batch_size 256
```


### Multiple machines executing within a local area network
**Ensure that the client and server can connect!**

Open the server terminal:
```shell
# Server Progress
python main.py --role server --arch vgg19_cifar10 --cutlayer deep \
--compressor ms --ratio 0.99 --bit 2 \
--client_ip client_ip --client_port 8000 --client_device cpu \
--server_ip server_ip --server_port 9000 --server_device cuda \
--batch_size 256
```

Open the client termianal:
```shell
# Client Progress
python main.py --role client --arch vgg19_cifar10 --cutlayer deep --path your_dataser_path \
--compressor ms --ratio 0.99 --bit 2 \
--client_ip client_ip --client_port 8000 --client_device cpu \
--server_ip server_ip --server_port 9000 --server_device cuda \
--batch_size 256
```
