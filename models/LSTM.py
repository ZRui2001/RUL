import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size1=50, hidden_size2=100, output_size=1, dropout=0.5, **kwargs):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # 设置 Dropout 层
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out