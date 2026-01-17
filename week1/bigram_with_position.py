
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy

# ==== 基本设置 ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ==== 读取文本 ====
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocabs = sorted(list(set(text)))
vocab_size = len(vocabs)
print("vocab_size:", vocab_size)

itos = {i: c for i, c in enumerate(vocabs)}
stoi = {c: i for i, c in enumerate(vocabs)}

encoding = lambda s: [stoi[c] for c in s]
decoding = lambda l: [itos[i] for i in l]

# ==== 划分训练 / 验证集 ====
n = int(len(text) * 0.9)
train_text = text[:n]
val_text = text[n:]

train_data = torch.tensor(encoding(train_text), dtype=torch.long)
val_data = torch.tensor(encoding(val_text), dtype=torch.long)

# ==== 超参数 ====
batch_size = 32
block_size = 8          # 最长上下文长度
n_embd = 32

#torch.manual_seed(1337)

# ==== 采样一个 batch ====
def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])          # (B, T)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])  # (B, T)
    x, y = x.to(device), y.to(device)
    return x, y

# ==== 模型定义 ====
class Bigram_Module(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        # idx: (B, T)
        B, T = idx.shape

        tok_emb = self.embedding_table(idx)                           # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )                                                              # (T, n_embd)
        x = tok_emb + pos_emb                                         # (B, T, n_embd)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        loss = None
        if target is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx: (B, T_start)
        for _ in range(max_new_tokens):
            # 只用最后 block_size 个 token 作为上下文，避免 position embedding 越界
            idx_cond = idx[:, -block_size:]          # (B, T_cond <= block_size)

            logits, _ = self(idx_cond)               # (B, T_cond, C)
            logits = logits[:, -1, :]                # (B, C)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx

# ==== 实例化模型 ====
m = Bigram_Module(vocab_size).to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

# ==== 评估函数 ====
eval_iters = 50

@torch.no_grad()
def estimate_loss():
    m.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = m(X, Y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    m.train()
    return out

# ==== 训练循环 ====
max_steps = 5000
eval_interval = 500
best_val_loss = math.inf
best_state = None
patience = 5
bad_count = 0

for step in range(1, max_steps + 1):
    x, y = get_batch('train')
    logits, loss = m(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}: train batch loss = {loss.item():.4f}")

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"=== eval at step {step}: train {losses['train']:.4f}, val {losses['val']:.4f} ===")

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            best_state = deepcopy(m.state_dict())
            bad_count = 0
            print(f"*** new best val loss {best_val_loss:.4f}, saving model ***")
        else:
            bad_count += 1
            print(f"no improvement, bad_count = {bad_count}")
            if bad_count >= patience:
                print("Early stopping triggered.")
                break

if best_state is not None:
    m.load_state_dict(best_state)
    print(f"Loaded best model with val loss {best_val_loss:.4f}")

# ==== 生成文本 ====
m.eval()
start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
max_new_tokens = 400
sampled = m.generate(start_idx, max_new_tokens)[0].tolist()
print("\n=== Generated text ===")
print(''.join(decoding(sampled)))
