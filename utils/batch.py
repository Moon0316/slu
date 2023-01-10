# coding=utf-8
import torch
from utils.vocab import PAD

def process_utt_for_LM(utt):
    '''
        在处理中文语言模型的输入时，有几个注意点：
        1. 中文的语言模型在输入大写英文字母时识别成[UNK]，需要把英文字母小写化
        2. 连续的小写字母也可能识别出错，需要把字母用空格分开
        3. 句子中可能包含了[PAD],[UNK]等token，需要保护这些token
    '''
    # 中文的语言模型在输入大写英文字母时识别成[UNK]，需要把
    utt = utt.replace(' ', '[PAD]')  # 空格视为PAD
    utt = utt.replace('[PAD]','@')  # 保护'[PAD]'
    utt = utt.replace('[UNK]','#')  # 保护'[UNK]'
    utt = ' '.join(list(utt.lower())) # 将英文转换成小写，然后加上空格
    utt = utt.replace('@','[PAD]')  # 恢复'[PAD]'
    utt = utt.replace('#','[UNK]')  # 恢复'[UNK]'
    
    return utt

def from_example_list(args, ex_list, device='cpu', train=True):
    ex_list = sorted(ex_list, key=lambda x: len(x.input_idx), reverse=True)
    batch = Batch(ex_list, device)
    pad_idx = args.pad_idx
    tag_pad_idx = args.tag_pad_idx
    batch.utt = [ex.utt for ex in ex_list]
     
    if args.pretrained_model:
        max_len = max([len(ex.utt) for ex in ex_list])
        batch.lm_utt = [ex.utt + ''.join([PAD]*(max_len-len(ex.utt))) for ex in ex_list]
        batch.lm_utt = [process_utt_for_LM(utt) for utt in batch.lm_utt]
             
    input_lens = [len(ex.input_idx) for ex in ex_list]
    max_len = max(input_lens)
    input_ids = [ex.input_idx + [pad_idx] * (max_len - len(ex.input_idx)) for ex in ex_list]    # 把每个句子的wordid用0补成batch中最长句子的长度
    batch.input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    batch.lengths = input_lens
    batch.did = [ex.did for ex in ex_list]

    if train:
        batch.labels = [ex.slotvalue for ex in ex_list]
        tag_lens = [len(ex.tag_id) for ex in ex_list]
        max_tag_lens = max(tag_lens)
        tag_ids = [ex.tag_id + [tag_pad_idx] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list]   # 把每个句子的tagid用0补成batch中最长句子的长度
        tag_mask = [[1] * len(ex.tag_id) + [0] * (max_tag_lens - len(ex.tag_id)) for ex in ex_list] # 设置mask，在模型前向传播的时候为0的位置不参与计算
        batch.tag_ids = torch.tensor(tag_ids, dtype=torch.long, device=device)
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)
    else:
        batch.labels = None
        batch.tag_ids = None
        tag_mask = [[1] * len(ex.input_idx) + [0] * (max_len - len(ex.input_idx)) for ex in ex_list]
        batch.tag_mask = torch.tensor(tag_mask, dtype=torch.float, device=device)

    return batch


class Batch():

    def __init__(self, examples, device):
        super(Batch, self).__init__()

        self.examples = examples
        self.device = device

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]