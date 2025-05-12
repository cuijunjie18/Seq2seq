import torch
from torchinfo import summary
from net_frame import *

# 超参数
batch_size,num_steps = 64,10
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

# 生成数据相关变量
data_iter,src_vocab,tgt_vocab = load_data_nmt(batch_size,num_steps)

# 搭建net
encoder = Seq2SeqEncoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
net = EncoderDecoder(encoder,decoder)

# 单序列推理测试
enc_x = torch.ones(1,10).to(torch.int32) # (batch,num_steps)
dec_x = torch.ones(1,10).to(torch.int32) # (batch,num_steps)
print(type(enc_x),enc_x.shape)
print(type(dec_x),dec_x.shape)
summary(net,input_data = (enc_x,dec_x))