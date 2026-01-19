import os
from typing import Optional, Union, Self

from sympy import transpose

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载 OpenMP
os.environ["OMP_NUM_THREADS"] = "1"          # 可选：限制线程数，避免过多线程

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int  = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = False




batch_size = 64
learning_rate = 3e-4


dropout = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device',device)

with open(r'C:\Users\Pillar\PythonProjects\week1\text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabs = sorted(list(set(text)))
vocab_size = len(vocabs)


stoi = {c:i for i,c in enumerate(vocabs)}
itos = {i:c for i,c in enumerate(vocabs)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text),dtype=torch.long).to(device)

# def get_batch(train_data, device):
#     ix = torch.randint(0, len(train_data) - block_size, (batch_size, ) )
#     x = torch.stack([train_data[i:i+block_size] for i in ix])
#     y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
#     x,y = x.to(device), y.to(device)
#     return x,y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class LayerNorm(nn.Module):
    def __init__(self, n_model,bias):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_model)) #乘积项系数
        self.beta = nn.Parameter(torch.zeros(n_model)) if bias else None #偏差
        #过程可以视作先将每一个token按分量做标准化，对于每个分量，给一个自己的gamma和beta


    def forward(self, x):
        return F.layer_norm(x,self.gamma.shape,self.gamma,self.beta,1e-6)

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed,3*config.n_embed,bias = config.bias)
        self.c_proj = nn.Linear(3*config.n_embed,config.n_embed,bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional,'scaled-dot_product_attention')

    def forward(self,x):
        B,T,C = x.shape
        q,k,v = self.c_attn(x).split(self.n_embed,dim = 2)
        q = q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)


        y = torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y










class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_size),
            wpe = nn.Embedding(config.block_size, config.embed_size),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            drop = nn.Dropout(config.dropout),
            ln_f = nn.LayerNorm(config.n_embed)
        ))
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj_weight'):
                torch.nn.init.normal_(p,mean=0,std=0.02/math.sqrt(config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_paramerters()/1e6),)

    def get_num_paramerters(self,non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params



    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0,std=0.02)


    #以下还未修改-----------------------------------------------------------------------------
    def forward(self,x,target):
        B,T = x.shape
        x = self.token_embedding_table(x)
        pos = self.position_embedding_table(torch.arange(T,device=x.device))
        x = x + pos

        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None

        if target is not None:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T,C)
            target_flat = target.view(B*T)
            loss = F.cross_entropy(logits_flat, target_flat)

        return logits,loss

    def generate(self,idx,max_token_size):

        for _ in range(max_token_size):
            idx_cond = idx[:,-block_size:]
            logits,_ = self.forward(idx_cond,None)
            logits_last = logits[:,-1,:]
            probs = F.softmax(logits_last,dim=-1)
            idx_next = torch.multinomial(probs,1)
            idx = torch.cat([idx,idx_next],dim=1)
        return idx

m = MiniGPT(vocab_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(),lr=learning_rate)

max_steps = 300

loss_figure = []


for step in range(max_steps):
    x,y = get_batch(data,device)
    logits,loss = m(x,y)
    loss_figure.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(step % 10 == 0):
        print(f"step: {step}, loss: {loss.item()}")

start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
max_new_tokens = 300
sampled = m.generate(start_idx, max_new_tokens)[0].tolist()
print(''.join(decode(sampled)))