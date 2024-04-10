#coding=utf-8
#import codecs

import jieba

import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset

from config import Config
from utils import load_json,dump_json,clean_special_chars


def create_vocab(data):
    '''
        args:data-->list:每条评论数据
    '''
    words_list = []
    for comment in data:
        comment = clean_special_chars(comment)
        words_list+=jieba.lcut(comment)
    vocab = Counter(words_list)
    del vocab['']
    del vocab[' ']
    return vocab

def load_vocab(config):
    vocab_dict = load_json(config.vocab_path)
    vocab_list = ['<pad>','<unknow>']
    vocab_list += [key for key,value in vocab_dict.items() if value >= config.vocab_min_cnt]
    
    text2idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx2text = {idx: word for idx, word in enumerate(vocab_list)}
    
    return text2idx,idx2text

def text2array(text,vocab,config):
    seq_len = config.seq_len
    text = clean_special_chars(text)
    words = jieba.lcut(text)
    if len(words) >= seq_len:
        words = words[:int(seq_len*0.5)]+words[(len(words)-int(seq_len*0.5)):]
        array = np.array([vocab.get(word,1) for word in words])
    else:
        array = np.array([vocab.get(word,1) for word in words])
        array = np.lib.pad(array, [0, config.seq_len - len(array)],
                          'constant',
                          constant_values=0)
    return array


class SentiDataSet(Dataset):
    def __init__(self,dataframe,vocab,config):
        super().__init__()
        self.comment = dataframe['comment'].apply(lambda x:text2array(x,vocab,config))
        self.label = dataframe['label']
        self.length = dataframe.shape[0]
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label_dict = {1:0,0:1,-1:2}
        comment = torch.from_numpy(self.comment[idx])
        label = self.label[idx]
        return comment, torch.tensor(label_dict[label])
    
if __name__ == '__main__':
    config = Config()
    data = pd.read_csv(config.train_path).fillna('')
    comment_list = data['comment'].to_list()
    vocab = create_vocab(comment_list)
    dump_json(vocab,config.vocab_path)   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        