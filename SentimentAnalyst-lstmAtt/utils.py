import time
import json
import pickle
import re
from datetime import timedelta
import datetime

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
    
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
    
def pkl_save(filename,file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()

def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file) 
    pkl_file.close()
    return file
    

def dump_json(vocab,file_path):
    with open(file_path,'w',encoding = 'utf-8') as file:
        vocab = json.dumps(vocab, ensure_ascii=False)
        file.write(vocab)
        
def load_json(file_path):
    with open(file_path,'r',encoding = 'utf-8') as file:
        vocab_dict = json.load(file)
    return vocab_dict
    
def clean_special_chars(text):
    p1 = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
    p2 = re.compile(r'\[.*?\]')
    text = re.sub(p1,'',text)
    text = re.sub(p2,'',text).replace('#','').replace('/n','').replace('#','').replace('@','')
    return text.lower() #字符转为小写

def data_split(data_,alpha):
    data = data_.drop_duplicates()
    train_iter_index = []
    active_iter_index = []
    for i in data.index:
        if (data.loc[i,'pre_dic'] == data.loc[i,'pre_model'])&(max(data.loc[i,'pre_g'],data.loc[i,'pre_m'],data.loc[i,'pre_b'])>alpha)&(data.loc[i,'CONTENT']!=''):
            train_iter_index.append(i)
        if ((data.loc[i,'pre_dic'] != data.loc[i,'pre_model'])&(max(data.loc[i,'pre_g'],data.loc[i,'pre_m'],data.loc[i,'pre_b'])>alpha))\
            or(max(data.loc[i,'pre_g'],data.loc[i,'pre_m'],data.loc[i,'pre_b'])<0.55)\
            or((data.loc[i,'pre_model'] == -1) and (data.loc[i,'pre_b'] > 0.8)):
            active_iter_index.append(i)
    train_iter = data.loc[train_iter_index,['CONTENT','pre_model']]
    train_iter.columns = ['comment','label']
    active_iter = data.loc[active_iter_index,['CONTENT','pre_model']]
    active_iter.columns = ['comment','label']
    return train_iter,active_iter

        