import pickle
import json
from net_frame import Vocab

def read_data(data_path):
    """读取json数据集，且tokenize"""
    with open(data_path,"r",encoding = 'utf-8') as file:
        data_set = json.load(file)

    source = []
    target = []
    for line in data_set:
        source.append(list(line['input'])) # 直接按字tokenize
        target.append(list(line['output'])) 
    return source,target

source,target = read_data(data_path = 'data/my_data.json')
reversed_tokens = ['<pad>','<bos>','<eos>']
src_vocab = Vocab(source,reserved_tokens = reversed_tokens)
tgt_vocab = Vocab(target,reserved_tokens = reversed_tokens)

# 保存词表
"""wb为二进制写入"""
with open('vocabs/src_vocab.pkl',"wb") as f:
    pickle.dump(src_vocab,f)
with open('vocabs/tgt_vocab.pkl',"wb") as f:
    pickle.dump(tgt_vocab,f)