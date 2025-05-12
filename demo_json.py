import json
import os

data_path = 'data/my_data.json'
with open(data_path,"r",encoding = 'utf-8') as file:
    data_set = json.load(file)

src_raw_text = []
tgt_raw_text = []
for line in data_set:
    src_raw_text.append(line['input'])
    tgt_raw_text.append(line['output'])

print(src_raw_text)
print(tgt_raw_text)