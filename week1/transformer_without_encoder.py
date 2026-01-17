
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from copy import deepcopy
#
# from torch.nn.functional import dropout
#
# # ==== 基本设置 ====
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using device:", device)
#
# # ==== 读取文本 ====
# with open('text.txt', 'r', encoding='utf-8') as f:
#     text = f.read()
#
# vocabs = sorted(list(set(text)))
# vocab_size = len(vocabs)
# print("vocab_size:", vocab_size)
#
# itos = {i: c for i, c in enumerate(vocabs)}
# stoi = {c: i for i, c in enumerate(vocabs)}
#
# encoding = lambda s: [stoi[c] for c in s]
# decoding = lambda l: [itos[i] for i in l]
#
# # ==== 划分训练 / 验证集 ====
# n = int(len(text) * 0.9)
# train_text = text[:n]
# val_text = text[n:]
#
# train_data = torch.tensor(encoding(train_text), dtype=torch.long)
# val_data = torch.tensor(encoding(val_text), dtype=torch.long)
#
# # ==== 超参数 ====
# batch_size = 64
# block_size = 256          # 最长上下文长度
# n_embd = 384
# n_head = 6
# n_layer = 6
# learning_rate = 3e-4
# dropout = 0.2
#
#
# torch.manual_seed(1337)
#
# # ==== 采样一个 batch ====
# def get_batch(split):
#     if split == 'train':
#         data = train_data
#     else:
#         data = val_data
#
#     ix = torch.randint(len(data) - block_size, (batch_size,))
#     x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T)
#     y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T)
#     x, y = x.to(device), y.to(device)
#     return x, y
#
# # ==== 模型定义 ====
# class Head(nn.Module):
#     def __init__(self, head_size):
#         super().__init__()
#         self.key = nn.Linear(n_embd, head_size)
#         self.value = nn.Linear(n_embd, head_size)
#         self.query = nn.Linear(n_embd, head_size)
#         # 因为 block_size 是全局常量，可以直接用
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         # 你这里原来有个 self.proj = nn.Linear(n_embd, n_embd)
#         # 但是在 forward 里没用到，可以先删掉，保持简洁
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x):
#         B, T, C = x.shape
#         k = self.key(x)              # (B, T, head_size)
#         q = self.query(x)            # (B, T, head_size)
#
#         # 注意缩放用 head_size，而不是 C（C 是 n_embd）
#         head_size = q.size(-1)
#         wei = q @ k.transpose(-2, -1) * (head_size ** -0.5)  # (B, T, T)
#
#         tril = self.tril[:T, :T]
#         wei = wei.masked_fill(tril == 0, float('-inf'))
#         wei = F.softmax(wei, dim=-1)
#         wei = self.dropout(wei)
#
#         v = self.value(x)            # (B, T, head_size)
#         out = wei @ v                # (B, T, head_size)
#         return out
#
# class MultiHeadAttention(nn.Module):
#     def __init__(self, num_heads,head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = nn.Linear(n_embd, n_embd)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self,x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.proj(out)
#         return out
# # 扩维度 + 非线性变换 + 压缩维度能表示复杂语义的原因：
# # 线性变换只能进行：投影、拉伸压缩...，无法改变拓扑结构；如果进行非线性变换，可以改变拓扑结构，形成线性变换无法达到的结构。
# # 高维空间能提供更多的可操作空间，因此先进行扩维。
# class FeedForward(nn.Module):
#     def __init__(self,n_embd):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(n_embd,4* n_embd),
#             nn.ReLU(),
#             nn.Linear(4*n_embd, n_embd),
#             nn.Dropout(dropout),
#         )
#
#     def forward(self,x):
#         return self.net(x)
#
# class Block(nn.Module):
#
#     def __init__(self, n_embd,n_head):
#         super().__init__()
#         head_size = n_embd // n_head
#         self.sa = MultiHeadAttention(n_head,head_size)
#         self.ffwd = FeedForward(n_embd)
#         self.ln1 = nn.LayerNorm(n_embd)
#         self.ln2 = nn.LayerNorm(n_embd)
#
#     def forward(self, x):
#         x = x+self.sa(self.ln1(x))
#         x = x+self.ffwd(self.ln2(x))
#         return x
#
#
#
# class Bigram_Module(nn.Module):
#     def __init__(self, vocab_size):
#         super().__init__()
#         self.embedding_table = nn.Embedding(vocab_size, n_embd)      # token embedding
#         self.position_embedding_table = nn.Embedding(block_size, n_embd)  # position embedding
#
#         self.blocks = nn.Sequential(
#             *[Block(n_embd,n_head = n_head) for _ in range(n_layer)],
#         )
#         self.ln_f = nn.LayerNorm(n_embd)
#         self.lm_head = nn.Linear(n_embd, vocab_size)  # 映射到 vocab 维度
#
#     def forward(self, idx, target=None):
#         # idx: (B, T)
#         B, T = idx.shape
#
#         tok_emb = self.embedding_table(idx)                  # (B, T, n_embd)
#         pos_emb = self.position_embedding_table(
#             torch.arange(T, device=idx.device)
#         )                                                    # (T, n_embd)
#         pos_emb = pos_emb.unsqueeze(0)                       # (1, T, n_embd) 方便广播
#         x = tok_emb + pos_emb                                # (B, T, n_embd)
#
#         x = self.blocks(x)                                   # (B, T, n_embd)
#         x = self.ln_f(x)                                     # (B, T, n_embd)
#         logits = self.lm_head(x)                             # (B, T, vocab_size)
#
#         loss = None
#         if target is not None:
#             B, T, C = logits.shape
#             logits_flat = logits.view(B * T, C)
#             target_flat = target.view(B * T)
#             loss = F.cross_entropy(logits_flat, target_flat)
#
#         return logits, loss
#
#     def generate(self, idx, max_new_tokens):
#         # idx: (B, T_start)
#         for _ in range(max_new_tokens):
#             # 只用最后 block_size 个 token 作为上下文，避免 position embedding 越界
#             idx_cond = idx[:, -block_size:]          # (B, T_cond <= block_size)
#
#             logits, _ = self(idx_cond)               # (B, T_cond, C)
#             logits = logits[:, -1, :]                # (B, C)
#             probs = F.softmax(logits, dim=-1)
#             idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
#             idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
#         return idx
#
# # ==== 实例化模型 ====
# m = Bigram_Module(vocab_size).to(device)
# optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
#
# # ==== 评估函数 ====
# eval_iters = 50
#
# @torch.no_grad()
# def estimate_loss():
#     m.eval()
#     out = {}
#     for split in ['train', 'val']:
#         losses = []
#         for _ in range(eval_iters):
#             X, Y = get_batch(split)
#             _, loss = m(X, Y)
#             losses.append(loss.item())
#         out[split] = sum(losses) / len(losses)
#     m.train()
#     return out
#
# # ==== 训练循环 ====
# max_steps = 1000
# eval_interval = 500
# best_val_loss = math.inf
# best_state = None
# patience = 5
# bad_count = 0
#
# for step in range(1, max_steps + 1):
#     x, y = get_batch('train')
#     logits, loss = m(x, y)
#
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
#     if step % 100 == 0:
#         print(f"step {step}: train batch loss = {loss.item():.4f}")
#
#     if step % eval_interval == 0:
#         losses = estimate_loss()
#         print(f"=== eval at step {step}: train {losses['train']:.4f}, val {losses['val']:.4f} ===")
#
#         if losses['val'] < best_val_loss:
#             best_val_loss = losses['val']
#             best_state = deepcopy(m.state_dict())
#             bad_count = 0
#             print(f"*** new best val loss {best_val_loss:.4f}, saving model ***")
#         else:
#             bad_count += 1
#             print(f"no improvement, bad_count = {bad_count}")
#             if bad_count >= patience:
#                 print("Early stopping triggered.")
#                 break
#
# if best_state is not None:
#     m.load_state_dict(best_state)
#     print(f"Loaded best model with val loss {best_val_loss:.4f}")
#
# # ==== 生成文本 ====
# m.eval()
# start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
# max_new_tokens = 1000
# sampled = m.generate(start_idx, max_new_tokens)[0].tolist()
# print("\n=== Generated text ===")
# print(''.join(decoding(sampled)))
