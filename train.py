import os
import time
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm

from model import *
from prepare import * 

# GPT Model Config
vocab_size = None
block_size = 1024 # Context Size, T
n_embd = 768
n_heads = 12
n_layers = 12
dropout = 0.2
# ------------

# Training Parameters
batch_size = 32 # Size of Batch, B
simulate_batching = True
mini_batch_size = 2
ddp = False
max_iters = 10000
eval_interval = 500 # To calculate an averaged out loss
save_interval = 500
learning_rate = 2.5e-4
min_learning_rate = 2.5e-5
warmup_iters = 2000
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
eval_iters = 500 # Average loss over 100 iters
# ------

DATASET_FOLDER = "./Data/"
MODEL_FOLDER = "./Model/"

if not os.path.exists(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

train_data = np.memmap(DATASET_FOLDER + "train.bin", dtype = np.uint16, mode = "r")
val_data = np.memmap(DATASET_FOLDER + "val.bin", dtype = np.uint16, mode = "r")

vocab_size = tokenizer.get_vocab_size()

config = GPTConfig(block_size, n_embd, n_heads, n_layers, dropout)

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(self, model, train_data, val_data, optimizer, rank):
        self.rank = rank
        self.model = model.to(self.rank)
        if ddp:
            self.model = DDP(self.model, device_ids = [self.rank])
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scaler = torch.cuda.amp.GradScaler()
        
    def get_batch(self, split, batch_size):
        # Generate one batch of training data of shape (batch_size, block_size)
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (batch_size,)) # batch_size many randint chunks of block_size
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix]) 
        y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


    def save_checkpoint(self, name):
        if ddp:
            checkpoint = self.model.module
        else:
            checkpoint = self.model
        torch.save(checkpoint, MODEL_FOLDER + name)

    @torch.no_grad()
    def evaluate_model(self, batch_size):
        # Generate eval_iter batches and calculate the average loss of the model on those batches
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                if simulate_batching:
                    X, Y = self.get_batch(split, mini_batch_size)
                else:
                    X, Y = self.get_batch(split, batch_size)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train() 
        return out

    def get_lr(self, iter):
        if iter < warmup_iters:
            return iter * learning_rate / warmup_iters
        decay = (iter - warmup_iters) / (max_iters - warmup_iters)
        return min_learning_rate + (0.5 * (1.0 + np.cos(np.pi * decay))) * (learning_rate - min_learning_rate)
        
    def run_iter(self):
        if simulate_batching:
            self.optimizer.zero_grad(set_to_none=True)
            for i in range(batch_size // mini_batch_size - 1):
                if ddp:
                    with self.model.no_sync():
                        x, y = self.get_batch('train', mini_batch_size)

                        logits, loss = self.model(x, y)
                        self.scaler.scale(loss).backward()
                else:
                    x, y = self.get_batch('train', mini_batch_size)

                    logits, loss = self.model(x, y)
                    self.scaler.scale(loss).backward()
            x, y = self.get_batch('train', mini_batch_size) # Sync the gradients

            logits, loss = self.model(x, y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            xb, yb = self.get_batch('train', batch_size)
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
    def train(self, max_iters):
        start = time.time()
        train_losses = []
        val_losses = []
        for iter in range(max_iters):
            if iter % eval_interval == 0: # Evaluate Loss
                losses = self.evaluate_model(batch_size)
                train_losses += [losses['train']]
                val_losses += [losses['val']]
                print(f"[GPU{self.rank}] Iteration {iter} | Batchsize: {batch_size}")
                print(f"Time taken {time.time()-start} secs, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                start = time.time()

            for param_group in self.optimizer.param_groups: # Set learning rate
                param_group['lr'] = self.get_lr(iter)

            self.run_iter()

            if iter % save_interval == 0 and (self.rank == 0 or self.rank == 'cuda'): # Save checkpoint
                self.save_checkpoint("checkpoint")
                print(f"[GPU{self.rank}] Iteration {iter} | Saving Checkpoint at {MODEL_FOLDER} ")

        losses = self.evaluate_model(batch_size)
        train_losses += [losses['train']]
        val_losses += [losses['val']]        
        # Evaluate and save the final model
        print(f"[GPU{self.rank}] Iteration {max_iters} | Batchsize: {batch_size}")
        print(f"Time taken {time.time()-start} secs, train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if self.rank == 0 or self.rank == 'cuda': 
            self.save_checkpoint('checkpoint')
            print(f"[GPU{self.rank}] Iteration {max_iters} | Saving Checkpoint at {MODEL_FOLDER} ")

        with open(MODEL_FOLDER + "train_losses.pkl", 'wb') as f:
            pickle.dump(train_losses, f)
        with open(MODEL_FOLDER + "val_losses.pkl", 'wb') as f:
            pickle.dump(val_losses, f)

def main(rank, world_size):
    if ddp:
        ddp_setup(rank, world_size)
    model = GPT(vocab_size, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, train_data, val_data, optimizer, rank)
    trainer.train(max_iters)
    if ddp: 
        destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Vocab Size: {vocab_size}")
    print(f"Number of Devices: {world_size}")
    if ddp:
        mp.spawn(main, args = (world_size,), nprocs = world_size)
    else:
        main(device, 1)
