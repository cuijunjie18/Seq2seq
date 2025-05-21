## Seq2seq

author：cuijunjie18

### 项目说明

训练一个端到端的机器翻译模型

### 模型说明

模型采用编码器——解码器架构.~~最终的训练效果较差~~

**效果差的原因**：predict函数中对输入的src_string处理与训练时不一致，训练为中文，按字tokenlize，而predict却是按word分，故修改net_frame中的predict函数，如下

```py
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, token = 'word',save_attention_weights=False): #@save
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    if token == 'word':
        src_tokens = src_vocab[src_sentence.lower().split(' ')] + [src_vocab['<eos>']] # 单词级别token
    else:
        src_tokens = src_vocab[list(src_sentence)] + [src_vocab['<eos>']]
```

即添加了一个指定数据处理的方式，token参数.

### 项目收获

- 学会使用python处理.json格式的数据集，学会使用json库
- 学会存储python中对象的方法，可以使用pickle库、joblib库的，具体两种使用在
  save_vocab1.py与save_vocab2.py
- 提高了pytorch使用熟练度，提高了对NLP简单任务整体流程的理解
- embedding,即嵌入层，是一种取代one_hot的好方法，尤其当vocab_size非常大时.
  输入(batch_size,step) -> (batch_size,step,embedding_size)