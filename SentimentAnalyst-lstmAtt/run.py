import torch
import pandas as pd
import numpy as np
import re

from torch.utils.data import DataLoader
from config import Config
from data_pro import load_vocab,SentiDataSet
from model import Model
from train_eval import model_train


def clean_special_chars(text):
    p1 = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
    p2 = re.compile(r'\[.*?\]')
    text = str(text)
    text = re.sub(p1,'',text)
    text = re.sub(p2,'',text).replace('#','').replace('/n','').replace('#','').replace('@','')
    return text.lower() #字符转为小写


def main():
    config = Config()
    word2idx,idx2word = load_vocab(config)
    train_data = pd.read_csv(config.train_path).fillna('').reset_index(drop = True)
    train_data['comment'] = train_data['comment'].apply(lambda x:clean_special_chars(x))
    train_dataset = SentiDataSet(train_data,word2idx,config)
    train_dataloader = DataLoader(train_dataset,config.batch_size,shuffle=True)
    valid_data = pd.read_csv(config.valid_path).fillna('').reset_index(drop = True)
    valid_dataset = SentiDataSet(valid_data,word2idx,config)
    valid_dataloader = DataLoader(valid_dataset,config.batch_size,shuffle = False)
    print('data  has been loaded!')
    
    model = Model(config)
    
    
    model_train(model,train_dataloader,valid_dataloader,config)
    
    
if __name__ == '__main__':
    main()