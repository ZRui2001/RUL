import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 获取序列的失效索引
def get_failure_idx(seq, threshold):
    indices = (seq <= threshold).nonzero()[0]
    if indices.size > 0:
        return indices[0]
    return float('nan')

# 读取数据：data_file(.npy) -> norm_sequence_data, failure_time
def read_and_norm(data_path, rated_capacity=1.1, failure_threshold=0.7):
    data = np.load(data_path, allow_pickle=True).item()
    # 标准化：对 capacity 数组整体除以额定容量
    norm_data = {battery_name: df['capacity'].values / rated_capacity for battery_name, df in data.items()}
    # 计算失效时间
    failure_time = {}  
    for battery, data in norm_data.items():
        failure_time[battery] = get_failure_idx(data, failure_threshold) + 1
    return norm_data, failure_time

# 数据划分：battery_data（字典） -> train_data, test_data
def split_data(battery_data, test_bat):
    train_data = {name: data for name, data in battery_data.items() if name != test_bat}
    test_data = battery_data[test_bat]
    return train_data, test_data

# 滑动窗口生成样本：sequence_data ->（sequences，labels）
def create_sequences(data, seq_length=64):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        label = data[i + seq_length]
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
def load_data(train_data, test_data, seq_length, batch_size):
    # 滑动窗口：生成序列数据样本和标签
    train_sequences, train_labels = [], []
    for _, data in train_data.items():
        sequences, labels = create_sequences(data, seq_length=seq_length)
        train_sequences.extend(sequences)
        train_labels.extend(labels)
    test_sequences, test_labels = create_sequences(test_data, seq_length=seq_length)

    # 创建 DataLoader
    train_dataset = TimeSeriesDataset(train_sequences, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TimeSeriesDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader