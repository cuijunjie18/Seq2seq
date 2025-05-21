import torch
import matplotlib.pylab as plt
import numpy as np
from net_frame import *
import json
import os
import joblib
from tqdm import tqdm

def get_index():
    with open("exp_num.json","r",encoding = 'utf-8') as file:
        content = json.load(file)
    return content['exp_num']

def update_index():
    with open("exp_num.json", "r+", encoding='utf-8') as file:
        content = json.load(file)
        index = content['exp_num']
        content['exp_num'] = index + 1
        file.seek(0)  # 回到文件开头
        file.truncate()  # 清空文件
        json.dump(content, file, ensure_ascii=False, indent=4)

def read_data(data_path):
    """读取json数据集，且tokenize"""
    with open(data_path,"r",encoding = 'utf-8') as file:
        dataset = json.load(file)

    source = []
    target = []
    for line in dataset:
        source.append(list(line['input'])) # 直接按字tokenize
        target.append(list(line['output'])) 
    return source,target

def load_train_data(batch_size,num_steps,data_path = 'data/my_data.json'):
    """构建train_iter"""
    source,target = read_data(data_path)
    # print(source)
    # print(target)
    reversed_tokens = ['<pad>','<bos>','<eos>']
    src_vocab = Vocab(source,reserved_tokens = reversed_tokens)
    tgt_vocab = Vocab(target,reserved_tokens = reversed_tokens)
    src_array,src_valid_len = build_array_nmt(source,src_vocab,num_steps)
    tgt_array,tgt_valid_len = build_array_nmt(target,tgt_vocab,num_steps)
    data_iter = load_array((src_array,src_valid_len,tgt_array,tgt_valid_len),batch_size)
    return data_iter,src_vocab,tgt_vocab

def train(net : nn.Sequential,data_iter : data.DataLoader,lr,tgt_vocab : Vocab,num_epochs = 10) -> list:
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    device = try_gpu()
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(),lr = lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    loss_plt = []
    for i in range(num_epochs):
        loss_epoch = 0
        tokens_nums = 0
        loop = tqdm(data_iter,total = len(data_iter))
        for batch in loop:
            optimizer.zero_grad()
            X,X_valid_len,Y,Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学(用当前时间步的实际标签作为decoder的输入),通常去掉最后<eos>，与label错开一个step
            Y_hat,_ = net(X,dec_input,X_valid_len)
            l = loss(Y_hat,Y,Y_valid_len)
            l.sum().backward()
            optimizer.step()

            # 累计损失
            loss_epoch += l.sum().item()
            tokens_nums += Y_valid_len.sum()
            loop.set_description(f'Epoch [{i + 1}/{num_epochs}]')
            loop.set_postfix({"LOSS" : loss_epoch / tokens_nums,"lr" : "{:e}".format(lr)})
        # if (loss_epoch / tokens_nums < 0.01):
        print("====================================================================")
        print(predict_seq2seq(net,"你是谁",src_vocab,tgt_vocab,num_steps,device,token = 'char')[0])
        print(predict_seq2seq(net,"什么是AI",src_vocab,tgt_vocab,num_steps,device,token = 'char')[0])
        loss_plt.append(loss_epoch / tokens_nums)
    return loss_plt

# 定义超参数
batch_size,num_steps = 1,20
embed_size, num_hiddens, num_layers, dropout = 128, 128, 4, 0
lr, num_epochs, device = 0.005, 100, d2l.try_gpu()

# 获取数据
train_iter,src_vocab,tgt_vocab = load_train_data(batch_size,num_steps)
# train_iter,src_vocab,tgt_vocab = load_data_nmt(batch_size,num_steps,data_path = 'en-cn/cmn.txt')
print(len(src_vocab),len(tgt_vocab))

# 定义net
encoder = Seq2SeqEncoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
# decoder = Seq2SeqAttentionDecoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout = dropout)
net = EncoderDecoder(encoder,decoder)

# # 检查随机性
# for batch in train_iter:
#     print("====================================")
#     X = [i.item() for x in batch[0] for i in x]
#     # Y = [i.item() for y in batch[2] for i in y]
#     print(src_vocab.to_tokens(X))
#     print(X)
#     # print(tgt_vocab.to_tokens(Y))
# src_sentence = "你是谁"
# X = src_sentence.lower().split(' ')
# print(X,type(X))
# predict_seq2seq(net,"你是谁",src_vocab,tgt_vocab,num_steps,device)[0]
# print(predict_seq2seq(net,"什么是AI",src_vocab,tgt_vocab,num_steps,device)[0])
# exit(0)

# 训练
loss_plt = train(net,train_iter,lr,tgt_vocab,num_epochs)

# 获取存储路径
index = get_index()
save_prefix = os.path.join('results','exp' + str(index))
os.makedirs(save_prefix,exist_ok = True)
os.makedirs(os.path.join(save_prefix,'model'),exist_ok = True)
os.makedirs(os.path.join(save_prefix,'vocabs'),exist_ok = True)

# 可视化结果
plt.plot(np.arange(len(loss_plt)),loss_plt)
plt.savefig(os.path.join(save_prefix,'loss_result.png'))
plt.show()

# 保存模型
save_path = os.path.join(save_prefix,'model','seq2seq.pt')
torch.save(net,save_path)
print(f"model save in {save_path} successfully!")

# 保存词表
joblib.dump(src_vocab,os.path.join(save_prefix,'vocabs','src_vocab.joblib'))
joblib.dump(tgt_vocab,os.path.join(save_prefix,'vocabs','tgt_vocab.joblib'))
# update_index()

# 检查随机性
for batch in train_iter:
    print("====================================")
    X = [i.item() for x in batch[0] for i in x]
    Y = [i.item() for y in batch[2] for i in y]
    print(src_vocab.to_tokens(X))
    print(tgt_vocab.to_tokens(Y))