#ATTENTION
import pandas as pd 
import torch 
batch_size = 4
block_size = 8
n_embd     = 32
n_head     = 4        # head_size = 8
n_layer    = 2
dropout    = 0.0      # à zéro tant que tu debugges
torch.manual_seed(1337)

# tinyshakespeare : le fichier input.txt de la vidéo
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)      # 65

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

print(head.train_data)