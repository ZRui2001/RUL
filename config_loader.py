"""
从配置文件config.yaml加载全局变量
"""

import yaml
import torch

# 加载配置文件
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 提取配置项
data_config = config['data']['CALCE']  # 仅使用 CALCE
models_config = config['models']
start_points = config['start_points']  # [0.3, 0.5, 0.7]
model_save_dir = config['model_save_dir']
seeds = config['seeds']

# 数据相关配置
data_path = data_config['data_path']
test_bat = data_config['test_bat']
seq_length = data_config['seq_length']
rated_capacity = data_config['rated_capacity']
failure_threshold = data_config['failure_threshold']

# 设备设置
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")