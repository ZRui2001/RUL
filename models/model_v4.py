"""
v1基础上，添加多尺度注意力
"""
import torch
import torch.nn as nn
import math

class model_v4(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1, **kwargs):
        """
        带动态多尺度注意力的 Transformer 模型。
        """
        super(model_v4, self).__init__()

        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.position_encoding = PositionalEncoding(model_dim, dropout)

        # 多尺度 Transformer 编码器
        self.layers = nn.ModuleList([
            MultiScaleTransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=model_dim * 4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

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

        # 转换为 Transformer 输入格式 (seq_length, batch_size, model_dim)
        x = x.permute(1, 0, 2)

        # 多尺度 Transformer 编码器
        for layer in self.layers:
            x = layer(x, seq_length)

        # 取最后一个时间步的输出
        x = x[-1]  # (batch_size, model_dim)

        # 输出预测
        return self.fc_out(x)  # (batch_size, output_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout=0.1, max_len=5000):
        """
        位置编码模块，使用正弦和余弦函数生成位置编码。
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe = torch.zeros(max_len, model_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 添加批量维度 (1, max_len, model_dim)
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

class MultiScaleAttention(nn.Module):
    def __init__(self, d_model, nhead):
        """
        多尺度注意力模块，支持动态设置局部窗口大小。
        参数:
        - d_model: 模型的嵌入维度
        - nhead: 多头注意力的头数
        """
        super(MultiScaleAttention, self).__init__()
        self.global_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)
        self.local_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

    def forward(self, x, local_window_size):
        """
        参数:
        - x: 输入序列，形状为 (seq_len, batch_size, d_model)
        - local_window_size: 动态计算的局部窗口大小
        
        返回:
        - 融合后的注意力结果，形状为 (seq_len, batch_size, d_model)
        """
        seq_len, batch_size, d_model = x.shape

        # 全局注意力
        global_out, _ = self.global_attention(x, x, x)

        # 局部注意力
        local_out = torch.zeros_like(x)
        for i in range(0, seq_len, local_window_size):
            start = max(0, i)
            end = min(seq_len, i + local_window_size)
            local_out[start:end], _ = self.local_attention(
                x[start:end], x[start:end], x[start:end]
            )

        # 融合全局和局部注意力
        return global_out + local_out


class MultiScaleTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(MultiScaleTransformerEncoderLayer, self).__init__()
        self.multi_scale_attention = MultiScaleAttention(d_model, nhead)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, seq_length):
        """
        参数:
        - src: 输入序列，形状为 (seq_len, batch_size, d_model)
        - seq_length: 输入序列长度，用于动态设置 local_window_size
        返回:
        - 输出序列，形状为 (seq_len, batch_size, d_model)
        """
        local_window_size = max(1, seq_length // 4)  # 动态设置窗口大小，最小为 1

        # 多尺度注意力
        src2 = self.multi_scale_attention(src, local_window_size)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.feedforward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


