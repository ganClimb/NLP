
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time
from torch.utils.data import DataLoader

path = os.getcwd()
sys.path.append(path+r'/utils')

from model_multilabel import Config,textDataSet,DPCNN,model_train,clean_special_chars,pkl_load,get_time_dif

def train(path,read_cols,map_label2num,num_epochs,batch_size,learning_rate,num_filters):
    remove_chas_path = path + r'\PreTrainModel\remove_chas_table.pkl'
    remove_chas_table = pkl_load(remove_chas_path)
    config = Config(path,read_cols,map_label2num,num_epochs,batch_size,learning_rate,num_filters)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    
    print('Preparing Divece...')
    print('='*20)
    if torch.cuda.is_available():    
        print('There are %d GPU(s) available.' % torch.cuda.validice_count())
        print('We will use the GPU:', torch.cuda.get_validice_name(0))
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
    print('='*50)

    start_time = time.time()
    print("Loading data...")
    print('='*50)
    #config = Config(dataSet)
    train_data = pd.read_csv(config.train_path,usecols = config.read_cols).fillna('').reset_index(drop = True)
    train_data['label'] = train_data['label'].apply(lambda x:x.replace(' ',''))
    train_data['content'] = train_data['TITLE']+train_data['DESCRIPTION']+train_data['KEYWORDS']
    train_data['content'] = train_data['content'].apply(lambda x:clean_special_chars(x,remove_chas_table))
    train_dataset = textDataSet(train_data,config,'content','label',config.map_label2num)
    print('='*20,'Traindata:','='*20)
    #train_dataset.dataset_static_info()
    train_dataloader = DataLoader(train_dataset,config.batch_size,shuffle=True)
    del train_data,train_dataset
    valid_data = pd.read_csv(config.valid_path,usecols = config.read_cols).fillna('').reset_index(drop = True)
    valid_data['label'] = valid_data['label'].apply(lambda x:x.replace(' ',''))
    valid_data['content'] = valid_data['TITLE']+valid_data['DESCRIPTION']+valid_data['KEYWORDS']
    valid_data['content'] = valid_data['content'].apply(lambda x:clean_special_chars(x,remove_chas_table))
    valid_dataset = textDataSet(valid_data,config,'content','label',config.map_label2num)
    print('='*20,'validdata:','='*20)
    #valid_dataset.dataset_static_info()
    valid_dataloader = DataLoader(valid_dataset,config.batch_size,shuffle=False)
    del valid_data
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print('='*50)
    
    # train
    model = DPCNN(config)
    #print(model.parameters)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    start_time = time.time()
    Epoch_acc,Epoch_loss,Epoch_loss_train,Epoch_acc_train = model_train(config, model, train_dataloader, valid_dataloader,start_time)
    
    # 作图查看loss 及 acc
    plt.figure(figsize = (8,5))
    f = pd.DataFrame(Epoch_acc,Epoch_loss).reset_index()
    f.columns=['loss','acc']
    plt.plot(f.index.to_list(),f.acc,color = 'red',lw = 1.0,ls=':', label='Accuracy_valid',marker = '+')
    plt.plot(f.index.to_list(),f.loss,color = 'blue',lw = 2.0,ls='--', label='Loss_valid',marker = '+')
    plt.xticks(rotation = 30,fontsize = 12,c = 'k') # rotation：旋转角度
    plt.yticks(rotation = 30,fontsize = 12,c = 'k')
    plt.grid(True)
    plt.legend(loc='upper right',edgecolor = 'none',facecolor = 'g',fontsize =13)
    plt.show()

if __name__ == '__main__':

    """配置参数 

    """
    #=======================================================================================================
    # 标签映射表
    map_label2num = {'孕期中':0,'妊娠纹':1,'婴童食品':2,'母婴辅食':3,
                     '洗护用品':4,'婴童穿搭':5,'亲子户外互动':6,
                     '身材修复':7,'亲子互动':8,'婴童出行':9,'记录母婴日常生活':10}
    # 样本字段
    read_cols = ['TITLE','DESCRIPTION','KEYWORDS','label']
    # 训练轮数
    num_epochs = 15
    # batch 大小
    batch_size = 512
    # 学习率
    learning_rate = 5e-3
    # CNN卷积核个数
    num_filters = 100
    #=======================================================================================================
    
    start_time = time.time()
    train(path,read_cols,map_label2num,num_epochs,batch_size,learning_rate,num_filters)