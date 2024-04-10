import torch
import pandas as pd
import sys
import os
from sklearn.metrics import classification_report

path = os.getcwd()
sys.path.append(path+r'/utils')
import model_multilabel as mm
from torch.utils.data import DataLoader
from model_multilabel import Config,textDataSet,DPCNN,clean_special_chars,evaluate,pkl_load


def Eval(path,model,Train = True,Valid = True,Test = True):
    
    remove_chas_path = path + r'\PreTrainModel\remove_chas_table.pkl'
    remove_chas_table = pkl_load(remove_chas_path)
    config = Config(path,read_cols,map_label2num,0,batch_size,0,100)
    print('='*50)
    if Train:
        train_data = pd.read_csv(config.train_path,usecols = config.read_cols).fillna('').reset_index(drop = True)
        train_data['content'] = train_data['KEYWORDS']+train_data['TITLE']+train_data['DESCRIPTION']
        train_data['content'] = train_data.content.apply(lambda x:clean_special_chars(x,remove_chas_table))
        train_dataset = textDataSet(train_data,config,'content',
                                    'label',config.map_label2num)
        train_dataloader = DataLoader(train_dataset,config.batch_size,shuffle=True)
        train_loss, train_acc, train_pre,train_recall,train_f1,train_hs, train_hl,labels_train,predict_train = evaluate(config, model,train_dataloader)
        msg = 'Train=== Loss: {0:>6.2}, Acc: {1:>6.2%}, Pre: {2:>6.2%}, Recall: {3:>6.2%} F1: {' \
                      '4:>6.2%} ,Hamming Score: {5:>6.2%}, Haimmng loss: {6:>6.2} '
        print(msg.format(train_loss, train_acc, train_pre, train_recall,train_f1,train_hs,train_hl))
        print(classification_report(labels_train, predict_train, digits=3))
    if Valid:
        valid_data = pd.read_csv(config.valid_path,usecols = config.read_cols).fillna('').reset_index(drop = True)
        valid_data['content'] = valid_data['KEYWORDS']+valid_data['TITLE']+valid_data['DESCRIPTION']
        valid_data['content'] = valid_data.content.apply(lambda x:clean_special_chars(x,remove_chas_table))
        valid_dataset = textDataSet(valid_data,config,'content',
                                  'label',config.map_label2num)
        valid_dataloader = DataLoader(valid_dataset,config.batch_size,shuffle=False)
        valid_loss, valid_acc, valid_pre,valid_recall,valid_f1,valid_hs, valid_hl,labels_valid,predict_valid = evaluate(config, model,valid_dataloader)
        msg = 'Valid=== Loss: {0:>6.2}, Acc: {1:>6.2%}, Pre: {2:>6.2%}, Recall: {3:>6.2%} F1: {' \
                      '4:>6.2%} ,Hamming Score: {5:>6.2%}, Haiming loss: {6:>6.2} '
        print(msg.format(valid_loss, valid_acc, valid_pre, valid_recall,valid_f1,valid_hs,valid_hl))
        print(classification_report(labels_valid, predict_valid, digits=3))
    if Test:
        test_data = pd.read_csv(config.test_path,usecols = config.read_cols).fillna('').reset_index(drop = True)
        test_data['content'] = test_data['KEYWORDS']+test_data['TITLE']+test_data['DESCRIPTION']
        test_data['content'] = test_data.content.apply(lambda x:clean_special_chars(x,remove_chas_table))
        test_dataset = textDataSet(test_data,config,'content',
                                  'label',config.map_label2num)
        test_dataloader = DataLoader(test_dataset,config.batch_size,shuffle=False)
        test_loss, test_acc, test_pre,test_recall,test_f1,test_hs,test_hl,labels_test,predict_test = evaluate(config, model,test_dataloader)
        msg = 'Test=== Loss: {0:>6.2}, Acc: {1:>6.2%}, Pre: {2:>6.2%}, Recall: {3:>6.2%} F1: {' \
                      '4:>6.2%} ,Hamming Score: {5:>6.2%}, Haiming loss: {6:>6.2} '
        print(msg.format(test_loss, test_acc, test_pre, test_recall,test_f1,test_hs,test_hl))
        print(classification_report(labels_test, predict_test, digits=3))

if __name__ == '__main__':
    
    """配置参数 

    """
    
    #============================================================================
    # bool值，True为需要对该训练集eval，False相反
    TRAIN = True
    VALID = True
    TEST = False

    # 标签映射表
    map_label2num = {'祛痘':0,'祛斑':1,'保湿补水':2,'疤痕修复':3,'美白':4,'好物分享测评':5,'护肤清单':6,'护肤前后':7,'脸部清洁':8}
    # 样本字段
    read_cols = ['TITLE','DESCRIPTION','KEYWORDS','label']
    # batch 大小
    batch_size = 512
    #============================================================================
    
    model = torch.load(path+r'/save_dict/DPCNN7_acc_0.928.pth')
    Eval(path,model,Train = TRAIN,Valid = VALID,Test = TEST)