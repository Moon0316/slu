# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np


class PointerNet(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, hidden_dim):
        super().__init__() 
        self.W1 = nn.Linear(encoder_dim, hidden_dim)
        self.W2 = nn.Linear(decoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_out):
        """
        :param encoder_out: (batch_size, seq_len, encoder_dim)
        :param decoder_out: (batch_size, decoder_dim)
        :return: (batch_size, seq_len), (batch_size)
        """

        hidden = self.activation(self.W1(encoder_out) + self.W2(decoder_out))
        attention0 = self.v(hidden).squeeze(2)
        attention = self.softmax(attention0)
        pointer = torch.argmax(attention, dim=1)
        
        return attention, pointer

class SLUPointer(nn.Module):

    def __init__(self, config, slot_vocab):
        super(SLUPointer, self).__init__()
        self.config = config
        self.negative_weight = config.negative_weight
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.encoder = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)    # 支持的类型包括LSTM，GRU，RNN
        # self.dropout_layer = nn.Dropout(p=config.dropout)
        self.slot_vocab = slot_vocab
        self.slot_embed = nn.Embedding(slot_vocab.num_tags, config.embed_size)
        self.decoder = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.pointer_net = PointerNet(config.hidden_size, config.hidden_size, config.ptr_size)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        # self.valid_weight = config.valid_weight


    def _forward_slot(self, batch, slot_id):
        input_ids = batch.input_ids
        lengths = batch.lengths

        embed = self.word_embed(input_ids)  # bsize x seqlen x vec_dim
        slot_embed = self.slot_embed(slot_id)  # bsize x vec_dim
        
        # calculate encoder output
        packed_inputs = rnn_utils.pack_padded_sequence(embed, lengths, batch_first=True, enforce_sorted=True)   # return a PackedSequence object
        packed_rnn_out, h_t_c_t = self.encoder(packed_inputs)  # bsize x seqlen x dim, h_t_c_t: (h_t, c_t), h_t: final_hidden_state, c_t: final_cell_state
        enoder_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_rnn_out, batch_first=True)
        # (h_e, c_e) = h_t_c_t    # h_e: bsize x num_layer*2 x dim, c_e: bsize x num_layer*2 x dim
        
        # calculate decoder output
        decoder_input = slot_embed.unsqueeze(1).expand(-1, 1, -1)  # bsize x 1 x dim
        
        # begin pointer     
        decoder_out, h_t_c_t = self.decoder(decoder_input, h_t_c_t)  # decoder_out: bsize x 1 x dim
        att_begin, ptr_begin = self.pointer_net(enoder_out, decoder_out)  # att: bsize x seqlen, pointer: bsize
        decoder_input = embed.gather(1, ptr_begin.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.embed_size))  # bsize x 1 x dim
           
        # end pointer
        decoder_out, h_t_c_t = self.decoder(decoder_input, h_t_c_t)  # decoder_out: bsize x 1 x dim
        att_end, ptr_end = self.pointer_net(enoder_out, decoder_out)  # att: bsize x seqlen, pointer: bsize
        decoder_input = embed.gather(1, ptr_end.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.config.embed_size))  # bsize x 1 x dim        
        
        return att_begin, att_end

    def forward(self,batch):
        len_slot_vocab = self.slot_vocab.num_tags
        batch_size = len(batch)
        max_len = batch.input_ids.shape[1]
        p_begins = torch.zeros((batch_size,len_slot_vocab,max_len), device=batch.input_ids.device)
        p_ends = torch.zeros((batch_size,len_slot_vocab,max_len), device=batch.input_ids.device)
        for slotid in range(len_slot_vocab):
            slotid_batch = torch.tensor([slotid]*batch_size, device=batch.input_ids.device)
            p_begin, p_end = self._forward_slot(batch, slotid_batch)
            p_begins[:,slotid,:] = p_begin
            p_ends[:,slotid,:] = p_end
            
        if batch.labels:    # calculate loss
            loss_begin = self.criterion(p_begins.view(-1,max_len), batch.slot_begin.view(-1)) \
                        * torch.tensor(np.where(batch.slot_begin.view(-1).cpu().numpy()>0.1, 1, self.negative_weight),device=batch.input_ids.device)
            loss_end = self.criterion(p_ends.view(-1,max_len), batch.slot_end.view(-1)) \
                        * torch.tensor(np.where(batch.slot_end.view(-1).cpu().numpy()>0.1, 1, self.negative_weight),device=batch.input_ids.device)
            loss = loss_begin.sum() + loss_end.sum()
            
            return p_begins, p_ends, loss
            
        return p_begins, p_ends

    def decode(self, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)    # p_begin,p_end,(loss)
        len_slot_vocab = self.slot_vocab.num_tags

        predictions = []
        for i in range(batch_size):
            pred_tuple = []
            for slot_id in range(len_slot_vocab):
                p_begin = output[0][i][slot_id,:]
                b_idx = torch.argmax(p_begin)
                p_end = output[1][i][slot_id,:]
                e_idx = torch.argmax(p_end)
                if b_idx==0 or e_idx==0:
                    continue
                else:
                    slot = self.slot_vocab.idx2tag[slot_id]
                    value = batch.utt[i][4:][b_idx:(e_idx+1)]   # '[PAD]sentence'
                    if value:   # 排除b_idx>e_idx的情况
                        pred_tuple.append(f'{slot}-{value}')

            predictions.append(pred_tuple)
            
        if len(output) == 2:    # test，没有label，因此没有loss
            return predictions
        else:
            loss = output[2]
            return predictions, labels, loss.cpu().item()
