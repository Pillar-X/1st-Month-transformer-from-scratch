import torch
import regex as re

with open(r"E:\Pillar\科研\Week3\taylorswift.txt",encoding = 'utf-8') as f:
    train_text = f.read()

#易错点：
# 1. bytes数据自始至终不会被直接处理，是先变成原始tokens，再替换tokens，比如(1,2) -> 300，把tokens_id为1，2的合成300
# 2. 最重要的是训练一个merges字典，记录(id1,id2) -> id3，方式是每次找到最大出现频率的pair，之后融合为新的id，修改原tokens序列；以此循环（可看出还需要两个函数get_stats和merge)
# 3. vocabs只是一个辅助作用，为了防止每次decode都重新算，可以在编制完merges之后，把每一个id对应的bytes串提前算好，之后只用把tokens换成bytes串就还原回去了
class BasicTokenizer():
    vocabs = {i:bytes([i]) for i in range(256)}
    merges = {}
    def __init__(self):
        pass

    def train(self,text,vocab_size):
        
        vocab_size0 = 256
        train_tokens = list(text.encode("utf-8"))
        for idx in range(vocab_size0,vocab_size):
            counts = self.get_stats(train_tokens)
            max_pair = max(counts,key = counts.get)
            self.merges[max_pair] = idx
            #max函数默认返回key,比较的时候也默认按key比较，除非主动更改比较的key，但返回的仍然是键
            train_tokens = self.merge(train_tokens,max_pair,idx)
        for (p0,p1),idx in sorted(self.merges.items(),key = lambda m: m[1]):
            self.vocabs[idx] = self.vocabs[p0] + self.vocabs[p1]
        print(list(self.merges.items()))

    def encode(self,text):
        tokens = list(text.encode('utf-8'))

        while(len(tokens)>1):
            stats = self.get_stats(tokens)
            min_pair = min(stats,key = lambda m : self.merges.get(m,float('inf')))
            if min_pair not in self.merges:
                break
            tokens = self.merge(tokens,min_pair,self.merges[min_pair])
        return tokens

    def decode(self,ids):
        tokens = b"".join(self.vocabs[x] for x in ids)
        text = tokens.decode('utf-8')
        return text

    def get_stats(self,tokens): # tokens: 字典id列表
        counts = {}
        for pair in zip(tokens,tokens[1:]):
            counts[pair] = counts.get(pair,0) + 1
        return counts

    def merge(self,tokens,pair,idx):
        new_tokens=[]
        i = 0
        while i < len(tokens):
            if i<len(tokens)-1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(idx)
                i += 2
            else :
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

bt = BasicTokenizer()
bt.train(train_text,3000)
# tokens = bt.encode('abc')
# texts = bt.decode(tokens)
# print(tokens)
# print(texts)


    

        
    