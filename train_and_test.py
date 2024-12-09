import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
from data_preprocess import *
from models.LSTM import LSTM
from models.GRU import GRU
from models.DeTransformer import DeTransformer
from models.model_v1 import model_v1
from models.model_v2 import model_v2
from models.model_v2_1 import model_v2_1
from models.model_v2_2 import model_v2_2
from models.model_v3 import model_v3
from models.model_v4 import model_v4
from models.model_v4_1 import model_v4_1
from models.model_v5 import model_v5
from typing import List, Dict


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_config, device, seq_length=None):
    model_name = model_config['name']
    if model_name == 'lstm':
        return LSTM(**model_config).to(device)
    elif model_name == 'gru':
        return GRU(**model_config).to(device)
    elif model_name == 'det':
        return DeTransformer(feature_size=seq_length, **model_config).to(device)
    elif model_name == 'model_v1':
        return model_v1(**model_config).to(device)
    elif model_name == 'model_v2':
        return model_v2(**model_config).to(device)
    elif model_name == 'model_v2_1':
        return model_v2_1(**model_config).to(device)
    elif model_name == 'model_v2_2':
        return model_v2_2(**model_config).to(device)
    elif model_name == 'model_v3':
        return model_v3(**model_config).to(device)
    elif model_name == 'model_v4':
        return model_v4(**model_config).to(device)
    elif model_name == 'model_v4_1':
        return model_v4_1(**model_config).to(device)
    elif model_name == 'model_v5':
        return model_v5(**model_config).to(device)

    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_optimizer(optim_name, model, lr, alpha=None):
    if optim_name == 'adam':
        return optim.Adam(model.parameters(), lr)
    elif optim_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr, alpha)
    else:
        raise ValueError(f"Unknown optimizer: {optim_name}")


def forward_prop(model_config, model, x):
    decodes = None
    if model_config['name'] == 'det':
        x = x.permute(0, 2, 1).repeat(1, model_config['feature_num'], 1)
        outputs, decodes = model(x)
    else:
        if x.shape[-1] == 64:
            print(x.shape)
            print(x)
        outputs = model(x)

    return outputs, decodes


def get_loss(model_config, model, sequences, labels, criterion):
    outputs, decodes = forward_prop(model_config, model, sequences)
    loss = criterion(outputs, labels.unsqueeze(-1))
    if model_config['name'] == 'det':
        loss += model_config['alpha'] * criterion(sequences.permute(0, 2, 1).repeat(1, model_config['feature_num'], 1),
                                                  decodes)
    return loss


def train_epoch(model_config, model, train_loader, device, optimizer, criterion):
    model.train()
    train_loss = 0
    for sequences, labels in train_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = get_loss(model_config, model, sequences, labels, criterion)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    return train_loss


def test_epoch(model_config, model, test_loader, device, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            test_loss += get_loss(model_config, model, sequences, labels, criterion).item()
    test_loss /= len(test_loader)
    return test_loss


def predict(model_config, model, sp, actual_seq, seq_length, failure_threshold, device):
    '''
    迭代预测

    输入：带参数的模型、预测点、真实值曲线

    输出：预测曲线（包含预测点前的真实值）
    '''
    model.eval()
    if actual_seq[-1] > failure_threshold:
        # B0007没有低于失效阈值的点
        actual_seq = np.append(actual_seq, failure_threshold - 1e-6)

    start_idx = int((get_failure_idx(actual_seq, failure_threshold) + 1) * sp)
    start_idx = max(start_idx, seq_length)
    with torch.no_grad():
        num_preds = len(actual_seq) - start_idx  # 让预测曲线与真实曲线同时结束
        preds = actual_seq[:start_idx]
        for _ in range(num_preds):
            input_seq = torch.tensor(preds[-seq_length:], dtype=torch.float32).to(device).reshape(1, -1, 1)
            pred, _ = forward_prop(model_config, model, input_seq)
            preds = np.append(preds, pred.item())
    return preds


def cal_metrics(actual_seq, pred_seq, sp, seq_length, failure_threshold=0.7):
    if actual_seq[-1] > failure_threshold:
        # B0007没有低于失效阈值的点
        actual_seq = np.append(actual_seq, failure_threshold - 1e-6)

    start_idx = int((get_failure_idx(actual_seq, failure_threshold) + 1) * sp)
    start_idx = max(start_idx, seq_length)
    # RE
    actual_failure_idx = get_failure_idx(actual_seq, failure_threshold)
    pred_failure_idx = get_failure_idx(pred_seq, failure_threshold)
    re = abs(actual_failure_idx - pred_failure_idx) / (actual_failure_idx + 1)
    if np.isnan(re):
        re = 1.000

    # 指标计算的时间边界，设为真实值的失效点
    end_idx = actual_failure_idx
    # RMSE
    rmse = np.sqrt(np.mean((actual_seq[start_idx:(end_idx + 1)] - pred_seq[start_idx:(end_idx + 1)]) ** 2))
    # MAE
    mae = np.mean(np.abs(actual_seq[start_idx:(end_idx + 1)] - pred_seq[start_idx:(end_idx + 1)]))
    return re, rmse, mae


def plot(actual_seq, pred_seqs: Dict[str, List[float]], sp, failure_threshold, seq_length, test_bat, figsize=(12, 6)):
    '''
    画多条预测曲线, 并自动分配不同颜色.

    参数:
        actual_seq: Actual value sequence.
        pred_seqs (Dict): Prediction result dictionary, with the following structure:
            - key (str): Name of model.
            - val (List): Prediction sequence, including actual values ahead start point.
    '''
    if actual_seq[-1] > failure_threshold:
        # B0007没有低于失效阈值的点
        actual_seq = np.append(actual_seq, failure_threshold - 1e-6)

    start_idx = int((get_failure_idx(actual_seq, failure_threshold) + 1) * sp)
    start_idx = max(start_idx, seq_length)
    colors = cm.viridis(np.linspace(0, 1, len(pred_seqs)))
    plt.figure(figsize=figsize)
    for (model_key, pred_seq), color in zip(pred_seqs.items(), colors):
        plt.plot(range(start_idx, len(pred_seq)), pred_seq[start_idx:], color=color, linestyle='--', linewidth=1.5,
                 label=model_key)  # 预测值曲线
    plt.plot(actual_seq, color='darkblue', linestyle='-', linewidth=1.5, label='Actual SOH')  # 真实值曲线
    plt.axhline(y=failure_threshold, color='red', linestyle='--', linewidth=2.5, label='Failure Threshold')  # 失效阈值线
    plt.axvline(x=start_idx, color='gray', linestyle='--', linewidth=1, label='Prediction Start Point')  # 预测起始点

    plt.xlabel('Cycles')
    plt.ylabel('SOH')
    plt.legend()
    plt.title(f'SOH degredation of {test_bat} (SP = {sp})')

    return plt


def eval_and_plot(model_names, model_paths, all_config):
    local_config = all_config[model_names[0]]
    failure_threshold = local_config['failure_threshold']
    test_battery_name = local_config['test_battery_name']
    pred_seqs = []
    for i in range(len(model_names)):
        actual_seq, pred_seq, start_idx = test(model_names[i], model_paths[i], all_config)

        # 计算RE、RMSE、MAE
        re, rmse, mae = cal_metrics(actual_seq, pred_seq, start_idx)
        print(f"Model:{model_names[i]}, RE: {re:.3f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        pred_seqs.append(pred_seq)

    plot(model_names, actual_seq, pred_seqs, start_idx, failure_threshold, test_battery_name)


def parse_model_filename(filename, pattern=r"exp-(\d+)_(\w+)_s-(\d+)\.pth"):
    match = re.match(pattern, filename)

    if match:
        exp_num = int(match.group(1))
        model_name = match.group(2)
        s_num = int(match.group(3))
        return exp_num, model_name, s_num
    raise ValueError(f"不符合预期格式的文件名：{filename}")
