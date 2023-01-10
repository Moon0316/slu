# coding=utf-8
import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator


def utt_manual_preprocess(string):
    utt = string
    del_list = ['(side)', '(dialect)', '(unknown)', '(noise)', '(robot)']
    for s in del_list:
        utt = utt.replace(s, '')
    if utt == '':
        utt = 'null'
    return utt


class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, use_manual=False):
        dataset = json.load(open(data_path, 'r', encoding='utf-8'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', use_manual)  # 这里会调用Example的初始化函数
                examples.append(ex)  # 插入的是一个example object，它包含的参数见下面的__init__函数
        return examples

    def __init__(self, ex: dict, did, use_manual=False):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did  # 作为每个句子在数据集中独一无二的标签

        if use_manual:
            self.utt = utt_manual_preprocess(ex['manual_transcript'])  # remove the 'unknown'
        else:
            self.utt = ex['asr_1best']
        self.slot = {}  # "act-slot": value
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}' # label[0]:act, label[1]:slot, label[2]:value
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)   # 初始化时认为每个字都属于O，后面修正
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value) # 找到value在utt中的起始位置
            if bidx != -1:  # 找到了
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]  # 获取utt中每个字在词库中的index（将输入标签化）
        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]  # 获取每个slot-act-value对在slot-act-value集合中的index（将输出标签化）
