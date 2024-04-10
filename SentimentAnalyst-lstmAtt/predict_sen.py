# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:35:15 2023

@author: Administrator
"""
# 加载相关包
import torch
import json
import re
import jieba
import numpy as np


label_dict = {0:1,1:0,2:-1}

def load_json(file_path):
    with open(file_path,'r',encoding = 'utf-8') as file:
        vocab_dict = json.load(file)
    return vocab_dict


def clean_special_chars(text):
    p1 = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
    p2 = re.compile(r'\[.*?\]')
    text = re.sub(p1,'',text)
    text = re.sub(p2,'',text.replace('\r','').replace('\s','')).replace('\f','').replace('\n','').replace('—','').replace('#','').replace('\t','').replace('#','').replace('@','').replace('	','')
    return text.lower() #字符转为小写

def load_vocab(vocab_path,vocab_min_cnt):
    vocab_dict = load_json(vocab_path)
    vocab_list = ['<pad>','<unknow>']
    vocab_list += [key for key,value in vocab_dict.items() if value >= vocab_min_cnt]
    
    text2idx = {word: idx for idx, word in enumerate(vocab_list)}
    idx2text = {idx: word for idx, word in enumerate(vocab_list)}
    return text2idx,idx2text

def getitem(text,vocab,seq_len):
    text = clean_special_chars(text)
    words = jieba.lcut(text)
    if len(words) >= seq_len:
        words = words[:int(seq_len*0.5)]+words[(len(words)-int(seq_len*0.5)):]
        array = np.array([vocab.get(word,1) for word in words])
    else:
        array = np.array([vocab.get(word,1) for word in words])
        array = np.lib.pad(array, [0, seq_len - len(array)],
                          'constant',
                          constant_values=0)
    return torch.from_numpy(array).reshape([1,50])

def model_predict(outputs):
    predic = torch.softmax(outputs,-1).detach().numpy()
    index_max = np.argmax(predic)
    res = label_dict[index_max]
    pre_g,pre_m,pre_b = predic[0],predic[1],predic[2]
    return res,pre_g,pre_m,pre_b

if __name__ == '__main__':
    comment = '猫:我和你说今天看到鬼了，又仔细看了一下，哦!原来是我那发癫的主人'
    model_path = r'E:\LK\NLPProject\SentimentAnalysis\sentiment-lstmAttCnn-master\save_dict_0228\24_acc_0.6124.pth'
    vocab_path = r'E:\LK\NLPProject\SentimentAnalysis\sentiment-lstmAttCnn-master\load_file\vocab.json'
    model = torch.load(model_path)
    model.eval()
    word2idx,idx2word = load_vocab(vocab_path,10)
    
    text_array = getitem(comment,word2idx,50)
    with torch.no_grad():
        res,pre_g,pre_m,pre_b = model_predict(model(text_array)[0])
        print(res,pre_g,pre_m,pre_b)
        
    import pandas as pd
    df = pd.read_csv(r'E:\LK\NLPProject\SentimentAnalysis\sentiment-lstmAttCnn-master\data\data_output\pre_sentiment2.csv',encoding = 'utf-8').fillna('')
    #df.columns = ['CONTENT']
    df['CONTENT'] = df.CONTENT.apply(lambda x:clean_special_chars(x))
    for i in df.index:
        text = getitem(df.loc[i,'CONTENT'],word2idx,50)
        res,pre_g,pre_m,pre_b = model_predict(model(text)[0])
        df.loc[i,'pre_model'] = res
        df.loc[i,'pre_g'] = pre_g
        df.loc[i,'pre_m'] = pre_m
        df.loc[i,'pre_b'] = pre_b
        
    
    df.to_csv(r'E:\LK\NLPProject\SentimentAnalysis\sentiment-lstmAttCnn-master\data\data_output\pre_sentiment2.csv',encoding = 'utf-8',index = False)
    
    
    
    
    
