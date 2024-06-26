import torch
import torch.nn as nn
import torch.nn.functional as F

class  Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.vocab_len, config.embedding_dim , padding_idx = 0)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        #self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size*2, config.num_classes)

    def forward(self, text):
        text = text.clone().detach().to(torch.int64)
        emb = self.embedding(text)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        #out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out