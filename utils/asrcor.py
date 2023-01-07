import sys, os, time, gc, json
import os
from args import init_args
import pycorrector

args = init_args(sys.argv[1:])
train_path = os.path.join(args.dataroot, 'train.json')
with open(train_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for diag in data:
    for dict in diag:
        sentence = dict['asr_1best']
        corrected_sent, detail = pycorrector.correct('少先队员因该为老人让坐')
        dict['asr_1best'] = corrected_sent

train_asrcor_path = os.path.join(args.dataroot, 'train_asrcor.json')
with open(train_asrcor_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)