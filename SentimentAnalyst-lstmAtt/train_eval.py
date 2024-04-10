import os
import torch
import numpy as np

import time

from sklearn import metrics
from torch.optim import AdamW
#from torch.utils.tensorboard import SummaryWriter
from transformers import  get_linear_schedule_with_warmup
from utils import get_time_dif


def APH(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred), \
           metrics.precision_score(y_true, y_pred, average='micro'), \
           metrics.recall_score(y_true, y_pred, average='micro'), \
           metrics.f1_score(y_true,y_pred,average='micro')




def model_predict(outputs):
    predic = torch.softmax(outputs,axis = 1)
    index_max = np.argmax(predic.detach().numpy(),axis = 1)
    return index_max


def model_train(model, train_dataloader, dev_dataloader,config):
    start_time = time.time()
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
    
    for epoch in range(config.num_epochs):
        timestamp = time.time()
        str_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        print("")
        print("Epoch Start Time:{}".format(str_time))
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.num_epochs))
        print('Training...')

        for i, (embedding,label) in enumerate(train_dataloader):
            model.train()
            optimizer.zero_grad()
            outputs = model(embedding).to(config.device)
            Loss = torch.nn.CrossEntropyLoss()
            loss = Loss(outputs, label.long())
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                pred = model_predict(outputs)
                #print(label,pred)
                train_acc, train_pre, train_recall,train_f1 = APH(label.data.numpy(), pred)
                dev_loss, dev_acc, dev_pre,dev_recall,dev_f1,labels_all,predict_all = model_evaluate(config, model,dev_dataloader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path+'_best'+'.pth')
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                
                msg = 'Iter: {0:>6}, Train=== Loss: {1:>6.2}, Acc: {2:>6.2%}, Pre: {3:>6.2%}, Recall: {4:>6.2%} F1: {' \
                      '5:>6.2%}  ;Val=== Loss: {6:>6.2}, Acc: {7:>6.2%}, Pre: {8:>6.2%}, Recall: {9:>6.2%} F1: {' \
                      '10:>6.2%}  Time: {11} {12} '
                print(msg.format(total_batch, loss.item(), train_acc, train_pre, train_recall,train_f1,
                                 dev_loss, dev_acc, dev_pre,dev_recall,dev_f1, time_dif, improve))
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
#         torch.save({'epoch': epoch, 'state_dict': model.state_dict(),
# 				'optimizer': optimizer.state_dict(),
# 				'scheduler': scheduler.state_dict()},
# 			    os.path.join(config.save_path, 'backbone_{}.pth'.format(epoch)))
        scheduler.step()  # 学习率衰减    
    return Epoch_acc,Epoch_loss,Epoch_loss_train,Epoch_acc_train


def model_evaluate(config, model, data_iter):
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for i,(texts, labels) in enumerate(data_iter):
            outputs = model(texts).to(config.device)
            Loss = torch.nn.CrossEntropyLoss()
            loss = Loss(outputs,labels.long())
            loss_total += loss
            labels = labels.data.cpu().numpy()
            pred = model_predict(outputs.data)
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, pred)
    #labels_all = labels_all.reshape(-1, config.num_classes)
    #predict_all = predict_all.reshape(-1, config.num_classes)
    acc, pre, recall,f1 = APH(labels_all, predict_all)
    
    return loss_total / len(data_iter),acc,pre,recall,f1,labels_all,predict_all