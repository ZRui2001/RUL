import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

def get_failure_idx(seq, threshold):
    """
    返回第一个小于等于阈值的索引，支持 NumPy 数组、Pandas Series 和 Python 列表。

    参数：
        seq: 可迭代对象 (NumPy 数组、Pandas Series、Python 列表等)
        threshold: 数值，用于比较的阈值

    返回：
        int 或 float: 满足条件的第一个索引；如果没有，返回 NaN
    """
    # 将输入统一转换为 NumPy 数组
    seq = np.asarray(seq)

    # 找到小于等于 threshold 的索引
    indices = (seq <= threshold).nonzero()[0]

    # 如果找到索引，返回第一个；否则返回 NaN
    if indices.size > 0:
        return indices[0]
    return float('nan')

def read_and_norm(data_path, rated_capacity, failure_threshold=0.7):
    '''
    读取、标准化数据文件为df, 并计算各电池失效时间 (dict).
    
    参数:
    data_path: 数据文件路径, csv文件
        需要包含字段: battery, cycle, capacity
    rated_capacity: 额定容量
    failure_threshold: SOH失效阈值
        
    返回:
    data_df: 标准化后的dataframe, 一行数据对应一个cycle
        列名:
        battery
        cycle
        capacity
        failure_cycle
    '''
    data_df = pd.read_csv(data_path, encoding='utf-8')
    data_df['capacity'] = data_df['capacity'] / rated_capacity

    # 计算失效时间
    batteries = data_df['battery'].unique()
    for battery in batteries:
        condition = data_df['battery'] == battery
        failure_cycle = get_failure_idx(data_df.loc[condition, 'capacity'], failure_threshold) + 1
        data_df.loc[condition, 'failure_cycle'] = failure_cycle    

    return data_df

def create_sequences(seq_arr, seq_length=64):
    '''
    滑动窗口生成样本

    参数:
    seq_arr: 多变量时间序列, 形状为 (num_cycles, num_features), 第一列为SOH

    返回:
    sequences: 多变量时间序列样本, 形状为 (样本个数, seq_length, num_features)
    labels: 标签 (SOH), 形状为 (样本个数,)
    '''
    sequences = []
    labels = []
    for i in range(len(seq_arr) - seq_length):
        sequence = seq_arr[i:i + seq_length]
        label = seq_arr[i + seq_length][0]
        sequences.append(sequence)
        labels.append(label)
    return sequences, labels

# （sequences，labels）-> Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, labels):
        if isinstance(sequences, list):  # 判断是否为列表类型
            sequences = np.array(sequences)  # 如果是列表，则转换为 numpy 数组，加速tensor的转化
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# dataloader：train_data(dict), test_data -> train_loader, test_loader
def load_data(batteries_df, val_bat, test_bat, seq_length, batch_size, features='capacity', use_failure_data=True):
    '''
    1. 划分数据集为训练、验证、测试
    2. 滑动窗口创造样本

    参数:
    batteries_df: 数据集
    val_bat: 验证集电池, 假设为单个
    test_bat: 测试集电池, 假设为单个
    seq_length: 滑动窗口大小
    features (str (单个) or array-like): 使用的特征, 容量在第一位

    返回:
    train_df, val_df, test_df, train_loader, val_loader, test_loader
    '''
    # 统一处理为一维array
    val_bat = np.atleast_1d(val_bat)
    test_bat = np.atleast_1d(test_bat)
    features = np.atleast_1d(features)
    
    train_sequences, train_labels = [], []
    val_sequences, val_labels = [], []
    test_sequences, test_labels = [], []
    train_data, val_data, test_data = [], [], []

    batteries = batteries_df['battery'].unique()
    for bat in batteries:
        condition = batteries_df['battery'] == bat
        battery_df = batteries_df[condition]
        seq_df = battery_df.loc[:, features]
        seq_arr = seq_df.to_numpy()  # shape: (num_cycles, num_features), 容量在第一列

        # 截去失效后数据（保留失效点）
        if not use_failure_data:
            failure_cycle = battery_df['failure_cycle'][0]
            seq_arr = seq_arr[:failure_cycle]

        # 滑动窗口：生成序列数据样本和标签
        seqs, labels = create_sequences(seq_arr, seq_length=seq_length)
        if bat == val_bat:
            val_sequences.extend(seqs)
            val_labels.extend(labels)
            val_data.append(battery_df)
        elif bat == test_bat:
            test_sequences.extend(seqs)
            test_labels.extend(labels)
            test_data.append(battery_df)
        else:
            train_sequences.extend(seqs)
            train_labels.extend(labels)
            train_data.append(battery_df)

    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)

    # 创建 DataLoader
    train_dataset = TimeSeriesDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TimeSeriesDataset(val_sequences, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TimeSeriesDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader