import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 允许重复加载 OpenMP
os.environ["OMP_NUM_THREADS"] = "1"          # 可选：限制线程数，避免过多线程


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


batch_size = 64
block_size = 256
embed_size = 384
head_num = 6
learning_rate = 3e-4
n_layer = 6
head_size = embed_size // head_num

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

def get_batch(train_data, device):
    ix = torch.randint(0, len(train_data) - block_size, (batch_size, ) )
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.key = nn.Linear(embed_size, head_size)
        self.value = nn.Linear(embed_size, head_size)
        self.query = nn.Linear(embed_size, head_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.last_attn = None  # <- 新增：存 attention

    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)      # (B, T, head_size)
        Q = self.query(x)
        V = self.value(x)

        wei = (Q @ K.transpose(-2, -1)) * (head_size ** -0.5)  # (B, T, T)
        tril = self.tril[:T, :T]
        wei = wei.masked_fill(tril == 0, float('-inf'))

        attn = F.softmax(wei, dim=-1)          # (B, T, T)
        self.last_attn = attn.detach()         # <- 关键：保存 dropout 前的 attention

        attn = self.dropout(attn)
        out = attn @ V                         # (B, T, head_size)
        return out


class MultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(head_num)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size*4),
            nn.ReLU(),
            nn.Linear(embed_size*4, embed_size),
            nn.Dropout(dropout),
    )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHead()
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        # x = self.ln1(self.sa(x) + x)
        # x = self.ln2(self.ffwd(x)+x)
        return x






class MiniGPT(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layer)],
        )
        self.ln_f = nn.LayerNorm(embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)


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


m.eval()
with torch.no_grad():
    # 取一段文本当作可视化输入（你也可以换成任意字符串）
    T_vis = 80  # 可视化的 token 数，太长标签会挤爆
    start = 0
    idx = data[start:start+T_vis].unsqueeze(0)  # (1, T)
    logits, _ = m(idx, None)

def plot_attn_heatmap(model, layer_id, head_id, idx, max_tokens=60):
    """
    layer_id: 0 ~ n_layer-1
    head_id:  0 ~ head_num-1
    idx: (1, T) 输入 token ids
    max_tokens: 画前多少个 token（标签太多会挤）
    """
    # 从模型里取出对应 head 的 attention
    head = model.blocks[layer_id].sa.heads[head_id]
    attn = head.last_attn  # (B, T, T)

    if attn is None:
        raise RuntimeError("last_attn 为空：请先 m.eval() 并跑一次 forward 再画图。")

    attn = attn[0]  # (T, T)
    T = attn.shape[0]
    T_show = min(T, max_tokens)

    # 取 token 字符做标签（你的是 char-level）
    token_ids = idx[0, :T_show].tolist()
    tokens = [itos[i] for i in token_ids]

    A = attn[:T_show, :T_show].detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(A, aspect='auto')
    plt.colorbar()
    plt.title(f"Attention Heatmap | Layer {layer_id} | Head {head_id}")

    plt.xticks(range(T_show), tokens, rotation=90, fontsize=8)
    plt.yticks(range(T_show), tokens, fontsize=8)
    plt.xlabel("Key position (被看)")
    plt.ylabel("Query position (谁在看)")
    plt.tight_layout()
    plt.show()


# 示例：画第 2 层第 4 个头
plot_attn_heatmap(m, layer_id=2, head_id=4, idx=idx, max_tokens=60)



start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
max_new_tokens = 300
sampled = m.generate(start_idx, max_new_tokens)[0].tolist()
print(''.join(decode(sampled)))

# # 保存loss曲线
# np.save("loss_curve_ffwd_times8.npy", np.array(loss_figure))
# # 保存loss曲线图像
# # ---- 绘图 ----
# plt.figure(figsize=(12, 7))
# plt.plot(loss_figure, label="Training Loss", color='b')
#
# plt.xlabel("Step", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.title("Training Loss Curve", fontsize=16)
# plt.grid(True)
#
# # ---- 添加超参数信息 ----
# text_info = (
#     f"batch_size = {batch_size}\n"
#     f"block_size = {block_size}\n"
#     f"embed_size = {embed_size}\n"
#     f"head_num = {head_num}\n"
#     f"n_layer = {n_layer}\n"
#     f"learning_rate = {learning_rate}\n"
#     f"dropout = {dropout}"
# )
#
# plt.text(
#     0.98, 0.95, text_info,
#     fontsize=11,
#     va='top', ha='right',
#     transform=plt.gca().transAxes,
#     bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", alpha=0.8)
# )
#
#
# plt.legend()
# plt.savefig("loss_curve_with_info_ffwd_times8.png", dpi=300)
# plt.show()