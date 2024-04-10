"""
Data Processing
1.The strategy of data annotation
2.Sampling
3.Spliting dataset
"""

import pandas as pd
import numpy as np
import os
import pickle
from tqdm.notebook import tqdm

path = os.getcwd()


def give_label(content,labels,hitword):
    '''
    params:content-->list 
           labels-->list
           hitword-->list
    return:res-->list
    '''
    res = []
    for i in tqdm(range(len(content))):
        label = []
        for j in range(len(labels)):
            a = 0
            for g in range(len(hitword[j])):
                if type(hitword[j][g]) == tuple:
                    if (hitword[j][g][0] in content[i]) and (hitword[j][g][1] in content[i]):
                        a+=1
                elif hitword[j][g] in content[i]:
                    a+=1
            if a != 0:
                label.append(labels[j])
        res.append(str(label).replace('[','').replace(']','').replace('\'',''))
        
    return res
    
def get_label_index(sample,labels):
    labels_num = len(labels)
    res = []
    for i in range(labels_num):
        index_list = []
        res.append(index_list)
    
    for i in sample.index:
        for j in range(labels_num):
            if labels[j] in sample.loc[i,'label']:
                res[j].append(labels[j])
    return res


    
    

def data_preprocessing(path,row_data_name,usecols,labels,hitwords,num_classes_samples = None,write = True):
    data_path = path + r'/data/data_raw' + raw_data_name
    data = pd.read_csv(data_path,usecols = usecols,encoding = 'utf-8').fillna('')
    data['content'] = data['TITLE']+data['DESCRIPTION']+data['KEYWORDS']
    print('data shape',data.shape)
    res_label = give_label(data.content.to_list(),labels,hitwords)
    data['label'] = res_label
    test = data.sample(5000)
    data.drop(test.index,axis = 0,inplace = True)
    
    res_index = get_label_index(data,labels)
    
    for x in range(len(res_index)):
        print(labels[x]+'命中样本量：',len(res_index[x]))
        
    train = pd.DataFrame()
    if num_classes_samples:
        for i in range(len(num_classes_samples)):
            train = pd.concat([train,data.loc[res_index[i],:].sample(num_classes_samples[i])],axis = 0)
            
    else:
        MIN = min([len(x) for x in res_index])
        for i in range(len(res_index)):
            if len(res_index[i])>MIN*5:
                train = pd.concat([train,data.loc[res_index[i],:].sample(MIN*5)],axis = 0)
            else:
                train = pd.concat([train,data.loc[res_index[i],:]],axis = 0)
    train.drop_duplicates(inplace = True)
    del data
    valid = train.sample(1000)
    train.drop(valid.index,inplace = True)
    
    if write:
        test.to_csv(path+r'/data/data_train/test.csv',index = False)
        print('Test dataset has been splited!')
        valid.to_csv(path+r'/data/data_train/valid.csv',index = False)
        print('Valid dataset has been splited!')
        train.to_csv(path+r'/data/data_train/train.csv',index = False)
        print(f'Train dataset has been splited! data size is {len(train):,}')
        
    else:
        return train,valid,test
            

if __name__ == "__main__":
    
    '''参数修改
    '''
    #==============================================================================================
    raw_data_name = ''
    
    # labels = ['孕期中','妊娠纹','婴童食品','母婴辅食',
    #           '洗护用品','婴童穿搭','亲子户外互动',
    #           '身材修复','亲子互动','婴童出行','记录母婴日常生活']
              
    # hitwords = [['孕期中','孕期日常','怀孕日记','孕期日记'],
    #             ['妊娠纹'],
    #             ['婴童食品'],
    #             ['宝宝辅食','母婴辅食','宝宝食谱',('自制','辅食')],
    #             ['洗护'],
    #             ['婴童穿搭','宝宝穿搭','萌娃穿搭','婴童时尚',('童装','婴童'),('萌娃','穿搭')],
    #             [('宝宝','户外'),('带娃','户外'),('儿童','户外'),('亲子','露营'),('亲子','户外'),('亲子','外景'),('亲子','出游'),('亲子','去哪儿'),('亲子','大自然')],
    #             ['身材修复','产后修复','产后恢复'],
    #             ['亲子游戏','亲子陪伴','亲子时光','亲子活动',('亲子','互动')],
    #             [('出行','宝宝'),('出行','带娃'),('旅行','宝宝'),('旅游','宝宝')],
    #             ['带娃','宝宝日常']]
    # usecols = ['ID','TITLE','DESCRIPTION','KEYWORdS']
    # #==============================================================================================            
    
    # data_preprocessing(path,raw_data_name,usecols,labels,hitwords,num_classes_samples = None,write = True)
    
                
    