import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout, **kwargs):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_sizes[0], batch_first=True)
        self.dropout = nn.Dropout(dropout)  # 设置 Dropout 层
        self.lstm2 = nn.LSTM(input_size=hidden_sizes[0], hidden_size=hidden_sizes[1], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out