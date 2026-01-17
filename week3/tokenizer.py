import torch
import regex as re

with open(r"E:\Pillar\大二上\科研\Week3\taylorswift.txt","r",encoding="utf-8") as f:
    train_text = f.read()

def get_stats(idx):
    # 输入token序列，返回字典 {pair : cnt}
    counts = {}
    for pair in zip(idx,idx[1:]):
        counts[pair] = counts.get(pair,0) + 1
    return counts

def merge(tokens,pair,idx):
    #输入原本的token序列，要合并的token和合并后的值，返回处理完成的新token序列
    i=0
    new_tokens = []
    while(i<len(tokens)):
        if( i<len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]):
            new_tokens.append(idx)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1
    return new_tokens


class BasicTokenizer:
    vocabs = {i : bytes([i]) for i in range(256)}
    merges = {}
    def __init__(self):
        pass

    def train(self,text,vocab_size,verbose=False):
        if(verbose == False):
            vocab_size0 = 256
            train_tokens = list(text.encode("utf-8"))
            #merge_size = vocab_size - vocab_size0
            for idx in range(vocab_size0,vocab_size):
                stats = get_stats(train_tokens)
                max_pair = max(stats,key=stats.get)
                self.merges[max_pair] = idx
                train_tokens = merge(train_tokens,max_pair,idx)
        print(list(self.merges.items()))
        #print(list(self.vocabs.items()))


    def encode(self,text):
        tokens = list(text.encode("utf-8"))

        while(len(tokens)>1):#如果tokens只有一个字符，不需要进行merges操作
            stats = get_stats(tokens)
            # 查询的对来自stats,但是要从merges中的idx来看从小到大操作
            min_pair = min(stats,key = lambda x: self.merges.get(x,float("inf")))
            if min_pair not in self.merges:
                break
            # 每次merge完，再看有没有新的能更新
            tokens = merge(tokens,min_pair,self.merges[min_pair])
        return tokens

    def decode(self,ids):
        for(p0,p1),idx in self.merges.items():
            self.vocabs[idx]  = self.vocabs[p0] + self.vocabs[p1]
            # vocabs 在0-255 为 idx: bytes([idx]) 在256以上是 idx:多个bytes拼接
        tokens = b"".join(self.vocabs[x] for x in ids)
        text = tokens.decode("utf-8")
        return text

bt = BasicTokenizer()
bt.train(train_text,400)
print(bt.encode("pillar"))
# print(bt.decode(bt.encode("llll")))

