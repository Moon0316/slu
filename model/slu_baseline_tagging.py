# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, BertModel


class SLUTagging(nn.Module):

    def __init__(self, config):
        super(SLUTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)    # 支持的类型包括LSTM，GRU，RNN
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)  # bsize x seqlen x vec_dim
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)   # return a PackedSequence object
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim, h_t_c_t: (h_t, c_t), h_t: final_hidden_state, c_t: final_cell_state
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)  # 将hidden_state通过一个全连接层，得到每个token的tag预测结果

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]    # output[0]: prob, output[1]: loss
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)   # idx->B/I-act-slot
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:]) # 对于BI...I，取B的act-slot为这个value的tag
                    value = ''.join([batch.utt[i][j] for j in idx_buff])    # i:batch中的第i个句子，j:第i个样本中的第j个token
                    idx_buff, tag_buff = [], [] # 清空
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:   # 处理最后一个tag
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:    # test，没有label，因此没有loss
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class SLUBertTagging(nn.Module):
    
    def __init__(self, config):
        super(SLUBertTagging, self).__init__()
        self.config = config
        self.cell = config.encoder_cell
        self.rnn = getattr(nn, self.cell)(768, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)    # 支持的类型包括LSTM，GRU，RNN
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        if config.pretrained_model == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.bertmodel = BertModel.from_pretrained("bert-base-chinese")
        elif config.pretrained_model == 'bertw':
            self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
            self.bertmodel = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        elif config.pretrained_model == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
            self.bertmodel = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        elif config.pretrained_model == 'macbert':
            self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base')
            self.bertmodel = BertModel.from_pretrained("hfl/chinese-macbert-base")
        
        self.device = config.device

    def forward(self, batch):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        utts = batch.lm_utt
        lengths = batch.lengths
        
        # convert text to bert input
        encoded_input = self.tokenizer(utts, return_tensors='pt', padding=True)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        bert_output = self.bertmodel(**encoded_input)
        # 开头字符是[CLS]，不考虑; 结尾字符是[SEP]，也不考虑
        embed = bert_output['last_hidden_state'][:,1:-1,:]
       
        # rnn
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)   # return a PackedSequence object
        packed_rnn_out, h_t_c_t = self.rnn(packed_inputs)  # bsize x seqlen x dim, h_t_c_t: (h_t, c_t), h_t: final_hidden_state, c_t: final_cell_state
        rnn_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)  # 将hidden_state通过一个全连接层，得到每个token的tag预测结果

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]    # output[0]: prob, output[1]: loss
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)   # idx->B/I-act-slot
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:]) # 对于BI...I，取B的act-slot为这个value的tag
                    value = ''.join([batch.utt[i][j] for j in idx_buff])    # i:batch中的第i个句子，j:第i个样本中的第j个token
                    idx_buff, tag_buff = [], [] # 清空
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:   # 处理最后一个tag
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:    # test，没有label，因此没有loss
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )
