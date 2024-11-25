"""
添加动态位置编码，加入时序信息
"""

import torch
import torch.nn as nn

class model_v2(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1, **kwargs):
        """
        用于时序预测的 Transformer 模型，支持滑动窗口添加动态位置编码。
        
        参数:
        - input_dim: 输入特征的维度
        - model_dim: 模型隐藏层维度 (Transformer embedding size)
        - num_heads: 多头注意力的头数
        - num_layers: Transformer 编码器的层数
        - output_dim: 输出维度 (通常为预测的时间步数量)
        - dropout: dropout 概率
        """
        super(model_v2, self).__init__()
        
        # 输入嵌入层
        self.input_embedding = nn.Linear(input_dim + 1, model_dim)
        
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
        
        # 动态位置编码
        position = torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).float() / seq_length
        position = position.unsqueeze(2).to(x.device)  # (batch_size, seq_length, 1)
        x = torch.cat([x, position], dim=2)  # (batch_size, seq_length, model_dim + 1)

        # 输入嵌入
        x = self.input_embedding(x)  # (batch_size, seq_length, model_dim)
        
        # 转换为 Transformer 的输入格式 (seq_length, batch_size, model_dim + 1)
        x = x.permute(1, 0, 2)
        
        # Transformer 编码器
        x = self.encoder(x)  # (seq_length, batch_size, model_dim + 1)
        
        # 取最后一个时间步的输出
        x = x[-1]  # (batch_size, model_dim + 1)
        
        # 输出预测
        output = self.fc_out(x)  # (batch_size, output_dim)
        return output