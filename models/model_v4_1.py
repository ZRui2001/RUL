"""
引入Inception模块，加入多尺度信息
"""

import torch
import torch.nn as nn
import math

class model_v4_1(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1, **kwargs):
        super(model_v4_1, self).__init__()
        # 线性嵌入层
        self.embedding = nn.Linear(input_dim, model_dim)
        self.position_encoding = PositionalEncoding(model_dim)

        # 后续 Inception-Transformer 层
        self.layers = nn.ModuleList([
            inception_transformer_block(
                input_dim=model_dim,  # 嵌入后的维度
                model_dim=model_dim,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

        # 全连接层
        self.fc = nn.Linear(model_dim, 1)  # 输出单步预测值

    def forward(self, x):
        # 线性嵌入
        x = self.embedding(x)  # [batch_size, seq_len, model_dim]

        # 添加位置编码
        x = self.position_encoding(x)

        x = x.permute(0, 2, 1)  # [batch_size, model_dim, seq_len]

        # 后续 Inception + Transformer 层
        for layer in self.layers:
            x = layer(x)

        # 取最后一个时间步
        x_last = x[:, :, -1]  # [batch_size, model_dim]
        x = self.fc(x_last)  # 输出 [batch_size, 1]
        return x

class inception_module(nn.Module):
    def __init__(self, input_dim, model_dim):
        super(inception_module, self).__init__()
        
        # 计算分支的通道数
        branch1_dim = model_dim // 4
        branch2_dim = model_dim // 2
        branch3_dim = model_dim // 8
        branch4_dim = model_dim - (branch1_dim + branch2_dim + branch3_dim)  # 分配剩余通道数
        
        # Branch 1: 1x1 Convolution
        self.branch1 = nn.Conv1d(input_dim, branch1_dim, kernel_size=1)
        
        # Branch 2: 1x1 → 3x3 Convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(input_dim, branch2_dim, kernel_size=1),
            nn.Conv1d(branch2_dim, branch2_dim, kernel_size=3, padding=1)
        )
        
        # Branch 3: 1x1 → 5x5 Convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(input_dim, branch3_dim, kernel_size=1),
            nn.Conv1d(branch3_dim, branch3_dim, kernel_size=5, padding=2)
        )
        
        # Branch 4: MaxPool → 1x1 Convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, branch4_dim, kernel_size=1)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate along channel dimension
        outputs = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        return outputs

class transformer_encoder_block(nn.Module):
    def __init__(self, model_dim, num_heads, dropout=0.1):
        super(transformer_encoder_block, self).__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.ReLU(),
            nn.Linear(model_dim * 4, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class inception_transformer_block(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, dropout=0.1):
        super(inception_transformer_block, self).__init__()
        self.inception = inception_module(input_dim=input_dim, model_dim=model_dim)
        self.transformer = transformer_encoder_block(model_dim=model_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, x):
        # Inception 模块
        inception_out = self.inception(x)  # [batch_size, model_dim, seq_len]
        # 转换为 Transformer 的输入格式
        inception_out = inception_out.permute(2, 0, 1)  # [seq_len, batch_size, model_dim]
        # Transformer 模块
        transformer_out = self.transformer(inception_out)
        return transformer_out.permute(1, 2, 0)  # 转回 [batch_size, model_dim, seq_len]

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
    
# # 参数定义
# input_dim = 1
# model_dim = 16
# seq_len = 16
# batch_size = 32

# # 初始化模型
# model = model_v4_1(
#     input_dim=input_dim,
#     model_dim=model_dim,
#     num_heads=4,
#     num_layers=3,
#     dropout=0.1
# )
# # 随机输入
# x = torch.rand(batch_size, seq_len, input_dim)  # [batch_size, input_dim, seq_len]

# # 模型输出
# output = model(x)
# print("模型输出形状:", output.shape)  # 应为 [batch_size, 1]