import torch
import torch.nn as nn
import math

class model_v1(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1, **kwargs):
        """
        用于时序预测的 Transformer 模型。
        
        参数:
        - input_dim: 输入特征的维度
        - model_dim: 模型隐藏层维度 (Transformer embedding size)
        - num_heads: 多头注意力的头数
        - num_layers: Transformer 编码器的层数
        - output_dim: 输出维度 (通常为预测的时间步数量)
        - dropout: dropout 概率
        """
        super(model_v1, self).__init__()
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.position_encoding = PositionalEncoding(model_dim, dropout)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=model_dim * 4, 
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        """
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, input_dim)
        
        返回:
        - 预测值，形状为 (batch_size, output_dim)
        """
        batch_size, seq_length, _ = x.shape
        
        # 输入嵌入与位置编码
        x = self.input_embedding(x)  # (batch_size, seq_length, model_dim)
        x = self.position_encoding(x)
        
        # 转换为 Transformer 的输入格式 (seq_length, batch_size, model_dim)
        x = x.permute(1, 0, 2)
        
        # Transformer 编码器
        x = self.encoder(x)  # (seq_length, batch_size, model_dim)
        
        # 取最后一个时间步的输出
        x = x[-1]  # (batch_size, model_dim)
        
        # 输出预测
        output = self.fc_out(x)  # (batch_size, output_dim)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        """
        位置编码模块，使用正弦和余弦函数生成位置编码。
        
        参数:
        - model_dim: Transformer 的隐藏层维度
        - dropout: dropout 概率
        - max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, model_dim)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
        - x: 输入张量，形状为 (batch_size, seq_length, model_dim)
        
        返回:
        - 加入位置编码后的张量，形状为 (batch_size, seq_length, model_dim)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)