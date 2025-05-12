import joblib
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

joblib.dump(src_vocab,'vocabs/src_vocab.joblib')
joblib.dump(tgt_vocab,'vocabs/tgt_vocab.joblib')