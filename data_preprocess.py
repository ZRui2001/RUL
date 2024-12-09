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
    """
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
    """
    data_df = pd.read_csv(data_path, encoding='utf-8')
    data_df['capacity'] = data_df['capacity'] / rated_capacity

    # 计算失效时间
    batteries = data_df['battery'].unique()
    for battery in batteries:
        condition = data_df['battery'] == battery
        failure_cycle = get_failure_idx(data_df.loc[condition, 'capacity'], failure_threshold) + 1
        data_df.loc[condition, 'failure_cycle'] = failure_cycle

    return data_df


def create_sequences(seq_arr, seq_length):
    """
    滑动窗口生成样本

    参数:
    seq_arr: 多变量时间序列, 形状为 (num_cycles, num_features), 第一列为SOH

    返回:
    sequences: 多变量时间序列样本, 形状为 (样本个数, seq_length, num_features)
    labels: 标签 (SOH), 形状为 (样本个数,)
    """
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


def get_loader(batteries_df, bats, seq_length, batch_size, shuffle, features='capacity', use_failure_data=True):
    if bats is None:
        return None, None

    bats = np.atleast_1d(bats)
    features = np.atleast_1d(features)

    x, y, data = [], [], []
    for bat in bats:
        condition = batteries_df['battery'] == bat
        bat_df = batteries_df[condition]
        seq_arr = bat_df.loc[:, features].to_numpy()

        if not use_failure_data:
            failure_cycle = bat_df['failure_cycle'].iloc[0]
            seq_arr = seq_arr[:failure_cycle]

        seqs, labels = create_sequences(seq_arr, seq_length=seq_length)
        x.extend(seqs)
        y.extend(labels)
        data.append(bat_df)

    data_df = pd.concat(data, ignore_index=True)

    dataset = TimeSeriesDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return data_df, dataloader


def load_data(batteries_df, test_bat, seq_length, batch_size, features='capacity', val_bat=None, use_failure_data=True):
    """
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
    """
    batteries = batteries_df['battery'].unique()
    train_bats = [bat for bat in batteries if bat not in (test_bat, val_bat)]

    train_df, train_loader = get_loader(batteries_df, train_bats, seq_length, batch_size, True, features,
                                        use_failure_data)
    val_df, val_loader = get_loader(batteries_df, val_bat, seq_length, batch_size, False, features, use_failure_data)
    test_df, test_loader = get_loader(batteries_df, test_bat, seq_length, batch_size, False, features, use_failure_data)

    return train_df, val_df, test_df, train_loader, val_loader, test_loader
