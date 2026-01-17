import torch
import torch.nn.functional as F
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('text.txt','r',encoding = 'utf-8') as F:
    text = F.read()

vocabs = sorted(list(set(text)))
vocab_size = len(vocabs)

itos = {i:c for i,c in enumerate(vocabs)}
stoi = {c:i for i,c in enumerate(vocabs)}
encoding = lambda s : [stoi[c] for c in s]
decoding = lambda l : [itos[i] for i in l]
#制定单个字符的index<->char表
#以及encoding:传入string->list[idx] 和 decoding: list[idx] -> string

n = int(len(text)*0.9)
train_data = text[:n]
vel_data = text[n:]
train_data = encoding(train_data)
train_data = torch.tensor(train_data,dtype = torch.long)
vel_data = torch.tensor(encoding(vel_data),dtype = torch.long)
#将train_data,vel_data转为tensor类型

batch_size = 32
block_size = 8
torch.manual_seed(1337)


def train_pair(trail):
    if trail == 'train':
        data = train_data
    else:
        data = vel_data

    select = torch.randint(len(data) - block_size, (block_size,))
    x = torch.stack([data[i:i + block_size] for i in select])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in select])  # ?
    x,y = x.to(device), y.to(device)
    return x, y
# x,y is (B,T)
# 调用的时候随机生成idx 和 target 张量

import torch.nn as nn
import torch.nn.functional as F
import torch


class Bigram_Module(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()  # 继承父类，这样才能保证optimizer等能正确找到参数，进行更新（已经在父类中包装好了）
        self.embedding_table = nn.Embedding(vocab_size, vocab_size)  # 模块如Embedding,Linear等需要在构造方法中写好
        # 这里Embedding只是相当于声明了一个vocab_size -> vocab_size的table

    def forward(self, idx, target=None):  # 目的：返回logits和计算loss
        logits = self.embedding_table(idx)
        B, T, C = logits.shape
        if target is None:  # 不能用target==None 来判定，否则会返回tensor而非boelean
            loss = None
        else:

            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
            # 交叉熵的传入格式：idx: (N,C) , target: (N,) 此方法会自动根据C代表的logits以及target给出的正确值，计算到答案

        return logits, loss

    def generate(self, idx, max_num):
        for _ in range(max_num):
            logits = self.embedding_table(idx)
            logits = logits[:, -1, :]  # 前面的概率用不着
            probs = F.softmax(logits, dim=1)
            idx_new = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_new), dim=1)
        return idx

m = Bigram_Module(vocab_size).to(device)

optimizer = torch.optim.AdamW(m.parameters(), 1e-1)

for _ in range(100):
        x, y = train_pair('train')
        logits, loss = m.forward(x, y)
        optimizer.zero_grad(set_to_none=True)  # 清空上一轮梯度值
        loss.backward()  # 内部自动计算梯度
        optimizer.step()  # 根据步长和梯度进行矩阵更新

print(loss)

idx = torch.zeros((1,1),dtype = torch.long,device = device)
max_num = 400
print(''.join(decoding(m.generate(idx,max_num)[0].tolist())))

