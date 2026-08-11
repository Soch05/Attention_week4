import torch 
import torch.nn as nn
import nn.functional as F


class HeadAttention(nn.Module):
    super().__init__()

    def __init__(self,n_emb, head_size, block_size, dropout = 0.0 ):
        # X (B, T, C = n_emb)
        # W = (head_size , C)
        # K = XWt --> (B,T, Head_size) B obtenu via broadcasting 
        self.key = nn.Linear(n_emb, head_size, biais : False) #(B,t,head_size)
        self.query = nn.Linear(n_emb, head_size, biais : False) #(B,t,head_size)
        self.val = nn.Linear(n_emb, head_size, biais : False) #(B,t,head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))
        self.dropout = nn.Dropout(dropout)
        



    def forward(self, x):
        B,T,C = x.shape
        self.K = key(x)
        self.Q = query(x)
        self.V = val(x)

        wei = Q @ K.transpose() * head_size ** -0.5#. (B,T,hs)@ (B,hs,T) -->( B,T,T)
        wei = wei.masked_fill(self.tril[ :T,:T ] == 0, float('-inf')
        wei = F.softmax(wei, dim = - 1)
        wei = self.dropout(wei)
        v = self.val(x)#(B,T,hs)
        out = wei @ v
