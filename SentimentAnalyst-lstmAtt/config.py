import os
import torch

class Config():
    path = os.getcwd()
    train_path = path+r'/data/data_train/train_0219.csv'
    valid_path = path+r'/data/data_train/valid.csv'
    test_path = path+r'/data/data_train/test.csv'
    vocab_path = path+r'/load_file/vocab.json'
    save_path = path+r'/save_dict_0228/'
    
    vocab_min_cnt = 10
    
    vocab_len = 71879
    embedding_dim = 512
    hidden_size = 256
    num_layers = 3
    dropout = 0.4
    num_classes = 3
    
    num_epochs = 25
    batch_size = 256
    learning_rate = 3e-5
    seq_len = 50
    require_improvement = 10000000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
if __name__ == '__main__':
    config = Config()
    