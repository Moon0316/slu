# coding=utf-8
import os, json
PAD = '[PAD]'
UNK = '[UNK]'
BOS = '<s>'
EOS = '</s>'


class Vocab():

    def __init__(self, padding=False, unk=False, min_freq=1, filepath=None):
        super(Vocab, self).__init__()
        self.word2id = dict()
        self.id2word = dict()
        if padding:
            idx = len(self.word2id)
            self.word2id[PAD], self.id2word[idx] = idx, PAD
        if unk:
            idx = len(self.word2id)
            self.word2id[UNK], self.id2word[idx] = idx, UNK

        if filepath is not None:
            self.from_train(filepath, min_freq=min_freq)

    def from_train(self, filepath, min_freq=1):
        with open(filepath, 'r', encoding='utf-8') as f:
            trains = json.load(f)
        word_freq = {}
        for data in trains:
            for utt in data:
                text = utt['manual_transcript'] # 通过manual_transcript统计词频
                for char in text:
                    word_freq[char] = word_freq.get(char, 0) + 1
        for word in word_freq:
            if word_freq[word] >= min_freq:
                idx = len(self.word2id)
                self.word2id[word], self.id2word[idx] = idx, word

    def __len__(self):
        return len(self.word2id)

    @property
    def vocab_size(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.word2id.get(key, self.word2id[UNK]) # 如果key不存在，返回UNK的id


class LabelVocab():

    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}

        self.tag2idx[PAD] = 0
        self.idx2tag[0] = PAD
        self.tag2idx['O'] = 1
        self.idx2tag[1] = 'O'
        self.ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r', encoding='utf-8'))
        self.from_filepath(root)

    def from_filepath(self, root):
        acts = self.ontology['acts']
        slots = self.ontology['slots']

        for act in acts:
            for slot in slots:
                for bi in ['B', 'I']:
                    idx = len(self.tag2idx)
                    tag = f'{bi}-{act}-{slot}'  # 遍历所有可能的BI，act，slot标签
                    self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)

class SlotVocab():
    
    def __init__(self, root):
        self.tag2idx, self.idx2tag = {}, {}
        self.from_filepath(root)

    def from_filepath(self, root):
        ontology = json.load(open(os.path.join(root, 'ontology.json'), 'r', encoding='utf-8'))
        acts = ontology['acts']
        slots = ontology['slots']

        for act in acts:
            for slot in slots:
                idx = len(self.tag2idx)
                tag = f'{act}-{slot}'  # 遍历所有可能的act，slot标签
                self.tag2idx[tag], self.idx2tag[idx] = idx, tag

    def convert_tag_to_idx(self, tag):
        return self.tag2idx[tag]

    def convert_idx_to_tag(self, idx):
        return self.idx2tag[idx]

    @property
    def num_tags(self):
        return len(self.tag2idx)