import os
import random
import time
import torch
import math
import numpy as np
import torch.nn as nn

from tqdm import tqdm

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

def vision_train(epoch, client, optimizer_c, server, optimizer_s, train_loader, device, comp, criterion):
    client.train()
    server.train()
    running_loss = 0.0
    total_batches = len(train_loader)

    start_time = time.time()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer_s.zero_grad()
        optimizer_c.zero_grad()

        smashed_data = client(data)
        with torch.no_grad():
            smashed_data_recover = comp.decode(*comp.encode(smashed_data)).to(device)
        smashed_data_recover.requires_grad = True
        
        output = server(smashed_data_recover)
        loss = criterion(output, target)
        loss.backward()

        smashed_data.backward(smashed_data_recover.grad)
        
        optimizer_c.step()
        optimizer_s.step()
        running_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / total_batches:.0f}%)]\tLoss: {loss.item():.6f}\tTime: {time.time() - start_time}')

    average_loss = running_loss / total_batches
    print(f'Epoch {epoch} Average Loss: {average_loss:.4f} Time: {time.time() - start_time}')
    return average_loss


def vision_validate(client, server, validate_loader, device, comp):
    client.eval()
    server.eval()
    total, correct1, correct5 = 0, 0, 0
    for data, target in validate_loader:
        data = data.to(device)
        target = target.to(device)
        total += len(data)

        smashed_data = client(data)
        smashed_data_recover = comp.decode(*comp.encode(smashed_data)).to(device) 
        output = server(smashed_data_recover)

        predict = output.argmax(dim=1)
        correct1 += torch.eq(predict, target).sum().float().item()
        target_resize = target.view(-1, 1)
        _, predict = output.topk(5)
        correct5 += torch.eq(predict, target_resize).sum().float().item()
    acc1 = correct1 / total
    acc5 = correct5 / total
    print(f'\nTop-1 Accuracy: {acc1}, Top-5 Accuracy: {acc5}\n')
    return acc1, acc5

def nlp_train(epoch, client, optimizer_c, server, optimizer_s, train_loader, device, comp, criterion, vocab_size):
    client.train()
    server.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer_c.zero_grad()
        optimizer_s.zero_grad()

        output = client(input_ids, attention_mask)
        smashed_data = output[0]
        with torch.no_grad():
            smashed_data_recover = comp.decode(*comp.encode(smashed_data)).to(device)
        smashed_data_recover.requires_grad = True
        output[0] = smashed_data_recover

        output = server(output)

        labels = nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        
        logits = output.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)

        loss = criterion(logits, shift_labels)
        
        loss.backward()
        smashed_data.backward(smashed_data_recover.grad)

        optimizer_c.step()
        optimizer_s.step()
        if not torch.isnan(loss):
            total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}")
    return total_loss / len(train_loader)
        
def nlp_validate(client, server, validate_loader, device, comp, criterion, vocab_size):
    client.eval()
    server.eval()
    total_eval_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(validate_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = client(input_ids, attention_mask)
            smashed_data = output[0]
            smashed_data_recover = comp.decode(*comp.encode(smashed_data)).to(device)
            output[0] = smashed_data_recover
            output = server(output)
            
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()
            
            logits = output.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(logits.device)

            loss = criterion(logits, shift_labels)

            if not torch.isnan(loss):
                total_eval_loss += loss.item()
    avg_eval_loss = total_eval_loss / len(validate_loader)
    perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 100 else float("inf")
    print(f'\nLoss: {avg_eval_loss}, PPL: {perplexity}\n')
    return avg_eval_loss, perplexity
