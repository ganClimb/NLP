""" config utf-8

"""
# 加载相关包
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
import time

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from torch.utils.data import Dataset
from torch.optim import AdamW
#from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel,BertTokenizer,BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm.notebook import tqdm
from datetime import timedelta, datetime


plt.rcParams['font.family'] = 'SimHei' # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

import warnings
warnings.filterwarnings('ignore')


def pkl_save(filename,file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()

def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file) 
    pkl_file.close()
    return file


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return sum([1 for i in range(len(labels_flat)) if pred_flat[i] == labels_flat[i]]) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def model_predict(outputs,alpha = 0.5,k = 7):
    '''
        params:output-->numpy.array 网络模型结果输出n*label_num
               alpha threshold
               k--topk
        return:pred-->tuple  [0]softmax,[1]预测结果n
    '''
    predic = torch.sigmoid(outputs)
    zero = torch.zeros_like(predic)
    topK = torch.topk(predic, k=k, dim=1, largest=True)[1]
    for i, x in enumerate(topK):
        for y in x:
            if predic[i][y] > alpha:
                zero[i][y] = 1
    return zero.cpu()

def clean_special_chars(text,remove_chas):
    for cha in remove_chas:
        text = text.replace(cha,'').replace('\n','').replace('#','').replace('话题','').replace(' ','').replace('[','').replace(']','')

    return text.lower() #字符转为小写


def report2df(report):
    output_list = []
    small_list = []
    list_report = report.split(' ')
    for i in list_report:
        if (' ' in i) or i == '':
            pass
        elif '\n' in i:
            small_list.append(i)
            output_list.append(small_list)
            small_list = []
        else:
            small_list.append(i)
    df = pd.DataFrame(output_list)
    return df

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0] )
        set_pred = set(np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)



def APH(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred), \
           metrics.precision_score(y_true, y_pred, average='samples'), \
           metrics.recall_score(y_true, y_pred, average='samples'), \
           metrics.f1_score(y_true,y_pred,average='samples'), \
           hamming_score(y_true,y_pred),\
           metrics.hamming_loss(y_true, y_pred)

# =====================================================================================
#                              1.初始化Config
# =====================================================================================
class Config(object):
    """配置参数"""
    def __init__(self,dataset_path,read_cols,map_label2num,num_epochs,batch_size,learning_rate,num_filters):
        self.dataset_path = dataset_path
        self.train_path = self.dataset_path + '/data/data_train/train.csv'                  # 训练集
        self.valid_path = self.dataset_path + '/data/data_train/valid.csv'                    # 验证集
        self.test_path = self.dataset_path + '/data/data_train/test.csv'                 # 测试集
        self.read_cols = read_cols
        self.map_label2num = map_label2num
        self.num_classes = len(map_label2num)
        self.map_num2label = {value:item for item,value in self.map_label2num.items()}                                           
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # 设备
        self.pretrained_path = str(dataset_path+'//PreTrainModel')
        self.embedding = BertModel.from_pretrained(self.pretrained_path).embeddings
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.require_improvement = 2500           # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = num_epochs                      # epoch数
        self.batch_size = batch_size                     # mini-batch大小
        self.pad_size = 512                       # 每句话处理成的长度(短填长切)
        self.learning_rate = learning_rate                 # 学习率
        self.embed = 768                          # embedding维度
        self.num_filters = num_filters
        self.model_name = 'DPCNN'
        self.save_path = dataset_path + '/save_dir/' + self.model_name   # 模型训练结果
        
        
        

        


# =====================================================================================
#                              2.定义Dataset
# =====================================================================================
# 加载数据集的目标是：1）把文本数据转化成预训练模型的词序、Mask 码，为输入进textCNN作准备；
#                  
class textDataSet(Dataset):
    """difine the dataset to training model 
    
    Arg:
        train_data: train_data-->pd.DataFrame
        config: Config-->Object
        return: word embedding vctor,label-->torch.Tensor
    """
    def __init__(self,train_data,config,content_name,label_name,map_table):
        super(textDataSet,self).__init__()
        self.config = config
        self.content_name = content_name
        self.label_name = label_name
        sentences,labels = self.load_data(train_data)
        self.labels = torch.tensor(self.Label2OneHot(labels,map_table))
        self.sentences = sentences
        self.map_table = map_table  #标签映射表 
        
        
        
    def load_data(self,train_data):
        return train_data[self.content_name],train_data[self.label_name]
    
    def Label2Num(self,label,maptable):
        label = label.split(',')
        for i,lb in enumerate(label):
            label[i] = maptable[lb]
        return label
    
    def Label2OneHot(self,labels,maptable):
        List = [[0 for i in range(self.config.num_classes)] for j in labels]
        for i, label in enumerate(labels):
            label_ = self.Label2Num(label,maptable)
            for lb in label_:
                List[i][lb] = 1
                #else:
                    #List[i][len(C) - 1] = 1 # 标签不在C中
        return List
    
    
    def dataset_static_info(self,config):
        count = 0
        for i, content in enumerate(self.sentences):
            token = config.tokenizer.tokenize(content) # 词列表
            seq_len = len(token)
            count += seq_len
        
        print(f"数据集总词数========{count}")
        print(f"数据集文本数========{len(self.sentences)}")
        print(f"数据集文本平均词数========{count / len(self.sentences)}")
        print(f"数据集样本平均标签数========{int(self.labels.sum()) / len(self.sentences)}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        # 对sentences进行编码处理
        content = self.sentences[idx]
        encoded_dict = self.config.tokenizer.encode_plus(
                content,                                 # 输入文本
                add_special_tokens=True,                 # 添加 '[CLS]' 和 '[SEP]'
                max_length=self.config.pad_size,         # 填充 & 截断长度
                pad_to_max_length=True,
                padding='max_length',
                truncation='only_first',
                return_attention_mask=True,              # 返回 attn. masks.
                return_tensors='pt'                      # 返回 pytorch tensors 格式的数据
            )
        return torch.squeeze(encoded_dict['input_ids'],0), self.labels[idx], torch.squeeze(encoded_dict['attention_mask'],0)

# =====================================================================================
#                              3.定义模型结构
# =====================================================================================
 

class  DPCNN(nn.Module):
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.embedding = config.embedding
        for param in self.embedding.parameters():
            param.requires_grad = False
        self.conv_region = nn.Conv2d(in_channels=1, out_channels=config.num_filters,
                                     kernel_size=(3, config.embed), stride=1)
        self.conv = nn.Conv2d(in_channels=config.num_filters, out_channels=config.num_filters,
                              kernel_size=(3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.padding1 = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.padding2 = nn.ZeroPad2d((0, 0, 0, 1))  # bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(config.num_filters, config.num_classes)
        #self.fc2 = nn.Linear(config.num_classes*5, config.num_classes)

    def forward(self,emb,mask):
        x = self.embedding(emb,mask) # [batch_size,seq_len,embeding_size]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embeding_size]
        x = self.conv_region(x)  # [batch_size, num_filters, seq_len-3+1, 1]

        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
		
        x = self.padding1(x)  # [batch_size, num_filters, seq_len, 1]
        x = self.relu(x)
        x = self.conv(x)  # [batch_size, num_filters, seq_len-3+1, 1]
        while x.size()[2] > 2:
            x = self._block(x)
        x = x.squeeze()  # [batch_size, num_filters(250)]
        x = self.fc(x)
        return x

    def _block(self, x):
        x = self.padding2(x)
        px = self.max_pool(x)

        x = self.padding1(px)
        x = F.relu(x)
        x = self.conv(x)

        x = self.padding1(x)
        x = F.relu(x)
        x = self.conv(x)

        # Short Cut
        x = x + px
        return x


# =====================================================================================
#                              4.定义训练函数、评估函数
# =====================================================================================  

def model_train(config, model, train_dataloader, dev_dataloader,start_time):
    
    params = list(model.named_parameters())
    print('The model has {:} different named parameters.\n'.format(len(params)))
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # BertAdam implements weight decay fix,
    # BertAdam doesn't compensate for bias as in the regular Adam optimizer.
    optimizer = AdamW(optimizer_grouped_parameters,lr=config.learning_rate,eps=1e-8)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = len(train_dataloader) * config.num_epochs)
    total_batch = 0               # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0              # 记录上次验证集loss下降的batch数
    flag = False                  # 记录是否很久没有效果提升
    Epoch_loss = []
    Epoch_acc = []
    Epoch_loss_train = []
    Epoch_acc_train = []
    
    for epoch in tqdm(range(config.num_epochs)):
        timestamp = time.time()
        str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        print("")
        print("Epoch Start Time:{}".format(str_time))
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.num_epochs))
        print('Training...')

        for i, (embedding,label,mask) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            outputs = model(embedding,mask).to(config.device)
            Loss = torch.nn.BCEWithLogitsLoss()
            loss = Loss(outputs, label.float())
            loss.backward()
            optimizer.step()
            if total_batch % 30 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                pred = model_predict(outputs,0.5,config.num_classes)
                train_acc, train_pre, train_recall,train_f1,train_hs,train_hl = APH(label.data.cpu().numpy(), pred.data.cpu().numpy())
                dev_loss, dev_acc, dev_pre,dev_recall,dev_f1,dev_hs, dev_hl,labels_all,predict_all = evaluate(config, model,dev_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path+'_best'+'.pth')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                
                msg = 'Iter: {0:>6}, Train=== Loss: {1:>6.2}, Acc: {2:>6.2%}, Pre: {3:>6.2%}, Recall: {4:>6.2%} F1: {' \
                      '5:>6.2%} ,Haiming Score: {6:>6.2%}, Haiming loss: {7:>6.2} ;Val=== Loss: {8:>6.2}, Acc: {9:>6.2%}, Pre: {10:>6.2%}, Recall: {11:>6.2%} F1: {' \
                      '12:>6.2%} ,Haiming Score: {13:>6.2%}, Haiming loss: {14:>6.2}, Time: {15} {16} '
                print(msg.format(total_batch, loss.item(), train_acc, train_pre, train_recall,train_f1,train_hs,train_hl,
                                 dev_loss, dev_acc, dev_pre,dev_recall,dev_f1,dev_hs, dev_hl, time_dif, improve))
                Epoch_loss.append(dev_loss)
                Epoch_acc.append(dev_acc)
                Epoch_loss_train.append(loss.item())
                Epoch_acc_train.append(train_acc)
        
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        
        report = metrics.classification_report(labels_all, predict_all, digits=3)
        print(report)
        torch.save(model,
            config.save_path +str(epoch)+ '_acc_'+str(dev_acc)+'.pth')# 每epoch保存一次模型
        scheduler.step()  # 学习率衰减
    torch.save(model,
               config.save_path +'final.pth')    
    return Epoch_acc,Epoch_loss,Epoch_loss_train,Epoch_acc_train


def evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for i,(texts, labels,mask) in enumerate(data_iter):
            outputs = model(texts,mask).to(config.device)
            Loss = torch.nn.BCEWithLogitsLoss()
            loss = Loss(outputs,labels.float())
            loss_total += loss
            labels = labels.data.cpu().numpy()
            pred = model_predict(outputs.data)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, pred.numpy())
    labels_all = labels_all.reshape(-1, config.num_classes)
    predict_all = predict_all.reshape(-1, config.num_classes)
    acc, pre, recall,f1,hs,hl = APH(labels_all, predict_all)
    
    return loss_total / len(data_iter),acc,pre,recall,f1,hs,hl,labels_all,predict_all
