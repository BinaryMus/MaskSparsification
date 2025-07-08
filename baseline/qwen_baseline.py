import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import math
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from datasets import load_dataset

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

device = 'cuda:0'

model_name = "Qwen/Qwen3-1.7B"
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

train_dataset = train_dataset.map(preprocess_function, batched=True, num_proc=20, remove_columns=["text"], load_from_cache_file=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True, num_proc=20, remove_columns=["text"], load_from_cache_file=True)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=20, collate_fn=default_data_collator, pin_memory=True)
eval_loader = DataLoader(eval_dataset, batch_size=4, num_workers=20, collate_fn=default_data_collator, pin_memory=True)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

def train(epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not torch.isnan(loss):
            total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    print(f"Epoch {epoch + 1}, Avg Loss: {total_loss / len(train_loader):.4f}")
    return total_loss / len(train_loader)

def eval():
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_eval_loss += outputs.loss.item()
    avg_eval_loss = total_eval_loss / len(eval_loader)
    perplexity = math.exp(avg_eval_loss) if avg_eval_loss < 100 else float("inf")
    return perplexity

loss_lst = []
perplexity_lst = []

perplexity = eval()
print(f"Validation Perplexity: {perplexity:.2f}")
perplexity_lst.append(perplexity)

for epoch in range(3):
    loss = train(epoch)
    perplexity = eval()
    loss_lst.append(loss)
    perplexity_lst.append(perplexity)
    print(f"Validation Perplexity: {perplexity:.2f}")

np.save('qwen_loss.npy', np.array(loss_lst))
np.save('qwen_perplexity.npy', np.array(perplexity_lst))