# coding=utf-8
import argparse
import sys


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt


def add_argument_base(arg_parser):
    #### General configuration ####
    arg_parser.add_argument('--dataroot', default='./data', help='root of data')
    arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
    arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    #### Training Hyperparams ####
    arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    #### Common Encoder Hyperparams ####
    arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='use for encoder and decoder')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')  # 表示word2vec时vec的维度
    arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')
    #### Our group's improvements ####
    arg_parser.add_argument('--pretrained_model', default=None, type=str, help='use which pretrained model for embedding, support bert|bertw|roberta|macbert,use word2vec if None')
    arg_parser.add_argument('--step_size', default=10, type=int, help='step_size of learning rate scheduler')
    arg_parser.add_argument('--gamma', default=0.5, type=float, help='gamma of learning rate scheduler')
    arg_parser.add_argument('--name', default='debug', type=str, help='experiment name')
    arg_parser.add_argument('--train_manual', action='store_true', help='train with manual_transcript and asr')
    arg_parser.add_argument('--hidden_size', default=512, type=int, help='encoder and decoder hidden size')
    arg_parser.add_argument('--ptr_size', default=128, type=int, help='pointer network hidden size')
    arg_parser.add_argument('--pointer_mode', default=True, type=bool, help='pointer mode,add extra token in utterance')
    arg_parser.add_argument('--negative_weight', default=0.02, type=float, help='loss weight of negative sample')
    arg_parser.add_argument('--do_correction', action='store_true', help='fine tune the output accoording to ontology')
    return arg_parser
