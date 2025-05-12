import torch
import os
from net_frame import *
import joblib

# 加载模型
index = 11
load_prefix = os.path.join('results','exp' + str(index))
model_path = os.path.join(load_prefix,'model','seq2seq.pt')
net = torch.load(model_path,weights_only = False)
# print(net)

# 加载词表
src_vocab = joblib.load(os.path.join(load_prefix,'vocabs/src_vocab.joblib'))
tgt_vocab = joblib.load(os.path.join(load_prefix,'vocabs/tgt_vocab.joblib'))
print(len(src_vocab),len(tgt_vocab))

# 构建对话
device = try_gpu()
num_steps = 20
input_str = "I admire you."
reply = predict_seq2seq(net,input_str,src_vocab,tgt_vocab,num_steps,device)[0]
print(reply)

# # en-cn的测试
# engs = ['I\'m free now.','I\'m innocent.','I\'m not sure.','I\'m pregnant.']	
# cns = ['我现在有空了。','我是清白的。','我不确定。','我怀孕了。']
# for eng,cn in zip(engs,cns):
#     translation, attention_weight_seq = predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device)
#     print(f'{eng} => {translation}, bleu {bleu(translation, cn, k=2):.3f}')

# # en-fra的测试
# engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# for eng, fra in zip(engs, fras):
#     translation, attention_weight_seq = predict_seq2seq(
#         net, eng, src_vocab, tgt_vocab, num_steps, device)
#     print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')