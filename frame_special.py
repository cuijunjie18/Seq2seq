from net_frame import *
import json

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
        for batch in data_iter:
            optimizer.zero_grad()
            X,X_valid_len,Y,Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                          device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学(用当前时间步的实际标签作为decoder的输入)
            Y_hat,_ = net(X,dec_input,X_valid_len)
            l = loss(Y_hat,Y,Y_valid_len)
            l.sum().backward()
            optimizer.step()

            # 累计损失
            loss_epoch += l.sum().item()
        loss_plt.append(loss_epoch)
        print(f"{i + 1}th epoch finish!")
    return loss_plt