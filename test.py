from frame_special import *

# 定义超参数
batch_size,num_steps = 64,10
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

# 获取数据
# train_iter,src_vocab,tgt_vocab = load_train_data(batch_size,num_steps)
train_iter,src_vocab,tgt_vocab = load_train_data(batch_size,num_steps)
print(len(src_vocab),len(tgt_vocab))

for batch in train_iter:
    X,X_valid_len,Y,Y_valid_len = batch
    # print(X,type(X),X.shape)
    # # print(src_vocab.to_tokens[X[0]])
    # print(X_valid_len,type(X_valid_len),X_valid_len.shape)
    # print(Y,type(Y),Y.shape)
    # print(Y_valid_len,type(Y_valid_len),Y_valid_len.shape)
    # # print(tgt_vocab.to_tokens[Y[0]])
    break

print(X_valid_len)
print(Y_valid_len)
str_src = []
str_tgt = []
for index in X:
    index = list(index)
    str_src.append(src_vocab.to_tokens(index))
for index in Y:
    index = list(index)
    str_tgt.append(tgt_vocab.to_tokens(index))

for x,y in zip(str_src,str_tgt):
    print(x,y)