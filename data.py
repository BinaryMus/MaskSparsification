from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from transformers import default_data_collator
from datasets import load_dataset

def cifar10(root='/dataset/cifar10', bs=256):
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
    train_set = CIFAR10(root=root, train=True, transform=transform_train, download=True)
    validate_set = CIFAR10(root=root, train=False, transform=transform_test)
    train_loader = DataLoader(train_set, bs, shuffle=True)
    validate_loader = DataLoader(validate_set, bs, shuffle=False)
    return train_loader, validate_loader

def cifar100(root='/dataset/cifar100', bs=256):
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
    train_set = CIFAR100(root=root, train=True, transform=transform_train, download=True)
    validate_set = CIFAR100(root=root, train=False, transform=transform_test)

    train_loader = DataLoader(train_set, bs, shuffle=True)
    validate_loader = DataLoader(validate_set, bs, shuffle=False)
    return train_loader, validate_loader


def wikitext2(tokenizer, bs=4):
    def preprocess_function(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )
        encodings["labels"] = encodings["input_ids"].clone()
        encodings["labels"][encodings["attention_mask"] == 0] = -100
        return encodings
    
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["text"], load_from_cache_file=True)
    eval_dataset = eval_dataset.map(preprocess_function, batched=True, remove_columns=["text"], load_from_cache_file=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, collate_fn=default_data_collator, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=bs, collate_fn=default_data_collator, pin_memory=True)
    
    return train_loader, eval_loader
