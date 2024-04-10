# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:35:15 2023

@author: Administrator
"""
# 加载相关包
import pandas as pd
import torch
import os
import sys

path = os.getcwd()
sys.path.append(path+r'/utils')
import model_multilabel as mm
from model_multilabel import Config,textDataSet,DPCNN,clean_special_chars,pkl_load
from transformers import BertTokenizer

def model_predict(config,output,alpha):
    res = []
    predic = torch.sigmoid(output).squeeze()
    for i in range(len(predic)):
        if predic[i] > alpha[i]:
            res.append((config.map_num2label[i],predic[i]))
    return str(res).replace('\'','')


def getitem(sample,config):
    if type(sample) == dict:
        if type(sample['TITLE']) != str:
            sample['TITLE'] = ''
        if type(sample['DESCRIPTION']) != str:
            sample['DESCRIPTION'] = ''
        if type(sample['KEYWORDS']) != str:
            sample['KEYWORDS'] = ''
        content = sample['TITLE']+sample['DESCRIPTION']+sample['KEYWORDS']
        
        
    elif type(sample) == list:
        if type(sample[1]) != str:
            sample[1] = ''
        if type(sample[2]) != str:
            sample[2] = ''
        if type(sample[3]) != str:
            sample[3] = ''
        content = sample[1]+sample[2]+sample[3]
        
    remove_chas_path = path + r'\PreTrainModel\remove_chas_table.pkl'
    remove_chas_table = pkl_load(remove_chas_path)
    content = clean_special_chars(content,remove_chas_table)
    encoded_dict = config.tokenizer.encode_plus(
                    content,                                 
                    add_special_tokens=True,                 
                    max_length=config.pad_size,        
                    pad_to_max_length=True,
                    padding='max_length',
                    truncation='only_first',
                    return_attention_mask=True,             
                    return_tensors='pt'                      
                )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']



if __name__ == "__main__":
    
    
    
    #=================================        样例         ==================================
    #字典格式
    sample1 = {'ID':'',
              'TITLE':'',
              'DESCRIPTION':'',   
              'KEYWORDS':''
              }
    
    #数组格式
    sample2 = [# ID
                '',  
               # TITLE
                '', 
               # DESCRIPTION 
                '',
               # KEYWORDS
                ''
              ]
    #输出概率阈值
    
    # map_label2num = {'孕期中':0,'妊娠纹':1,'婴童食品':2,'母婴辅食':3,
    #                  '洗护用品':4,'婴童穿搭':5,'亲子户外互动':6,
    #                  '身材修复':7,'亲子互动':8,'婴童出行':9,'记录母婴日常生活':10}
    
    # alpha = [0.9 for x in range(len(map_label2num))]
    # #alpha = [0.5,0.5,0.5,0.9,0.99,0.99,0.99]
    # #=======================================================================================
    
    # config = Config(path,'',map_label2num,0,0,0,0)
    # model=torch.load(path+r'/save_dict/DPCNNfinal.pth')
    # model.eval()
    # text,mask = getitem(sample1,config)
    # with torch.no_grad():    
    #     res = model_predict(config,model(text,mask),alpha =alpha)
    #     print(res)
        
    #     df = pd.read_csv(path+r'/data/data_train/test.csv').reset_index(drop = True).fillna('').sample(1000)
    #     for i in df.index:
    #         sample = ['',df.loc[i,'TITLE'],df.loc[i,'DESCRIPTION'],df.loc[i,'KEYWORDS']]
    #         text,mask = getitem(sample,config)
    #         df.loc[i,'pre'] = model_predict(config,model(text,mask),alpha = alpha)
    #     df[['ID','KEYWORDS','pre']].to_csv(path+r'/data/data_output/pre.csv',index = False)
  

    