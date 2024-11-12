import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from datetime import datetime
from data_preprocess import load_data, read_and_norm, get_failure_idx
from models import LSTM, GRU, DeTransformer
import matplotlib.pyplot as plt

def get_model_and_optim(model_name, all_config):
    local_config = all_config[model_name]
    if model_name == 'LSTM':
        model = LSTM(**local_config)
        optimizer = optim.RMSprop(model.parameters(), local_config['lr'], local_config['alpha'])
    elif model_name == 'GRU':
        model = GRU(**local_config)
        optimizer = optim.Adam(model.parameters(), local_config['lr'])
    elif model_name == 'DeTransformer':
        model = DeTransformer(**local_config)
        optimizer = optim.Adam(model.parameters(), local_config['lr'])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.to(device=local_config['device'])
    return model, optimizer

def forward_prop(model_name, model, x, all_config):
    local_config = all_config['DeTransformer']
    decodes = None
    if model_name == 'DeTransformer':
        x = x.unsqueeze(-1).permute(0, 2, 1).repeat(1, local_config['K'], 1)
        outputs, decodes = model(x)
    else:
        outputs = model(x.unsqueeze(-1))

    return outputs, decodes

def get_loss(model_name, model, x, y, criterion, all_config):
    local_config = all_config['DeTransformer']
    outputs, decodes = forward_prop(model_name, model, x, all_config)
    loss = criterion(outputs, y.unsqueeze(-1))
    return loss if decodes is None else loss + local_config['alpha'] * criterion(
        x.unsqueeze(-1).permute(0, 2, 1).repeat(1, local_config['K'], 1), decodes
        )

# 训练
def train(model_name, all_config):
    config = all_config[model_name]
    num_epochs = config['num_epochs']
    device = config['device']
    lr = config['lr']
    batch_size = config['batch_size']

    # 数据准备
    train_loader, test_loader = load_data(**config)

    # 获取初始化的模型、优化器
    model, optimizer= get_model_and_optim(model_name, all_config)
    criterion = nn.MSELoss()

    # 格式化当前日期时间为 "YYMMDD_HHMM"
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    # 训练循环
    train_losses = []
    test_losses = []
    best_test_loss = float('inf')  # 初始化一个较大的损失值
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = get_loss(model_name, model, sequences, labels, criterion, all_config)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)  # 记录训练损失

        # 测试
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                test_loss += get_loss(model_name, model, sequences, labels, criterion, all_config).item()
        test_loss /= len(test_loader)
        test_losses.append(test_loss)  # 记录测试损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}")

        # 保存测试损失最低的模型
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            dir = f'checkpoints/{model_name}_{timestamp}_lr-{lr}_bs-{batch_size}'
            os.makedirs(dir, exist_ok=True)
            torch.save(model.state_dict(), f'{dir}/epc-{epoch+1}.pth')
            print(f"New best model saved")

def predict(model_name, model, start_idx, actual_seq, window_size, device, all_config):
    with torch.no_grad():
        num_preds = len(actual_seq) - start_idx  # 让预测曲线与真实曲线同时结束
        preds = actual_seq[start_idx - window_size:start_idx]
        for _ in range(num_preds):
            input_seq = torch.tensor(preds[-window_size:], dtype=torch.float32).to(device).unsqueeze(0)
            pred, _ = forward_prop(model_name, model, input_seq, all_config)
            preds = np.append(preds, pred.item())
    return preds[window_size:]

def test(model_name, model_path, all_config):
    # 超参数
    config = all_config[model_name]

    # 加载模型
    model, _ = get_model_and_optim(model_name, all_config)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置模型为评估模式

    # 获取真实值序列
    battery_data, failure_time = read_and_norm(config['data_path'])
    actual_seq = battery_data[config['test_battery_name']]

    # 获取预测起始点，迭代预测
    start_idx = int(failure_time[config['test_battery_name']] * config['start_point'])
    pred_seq = predict(model_name, model, start_idx, actual_seq, config['window_size'], config['device'], all_config)

    return actual_seq, pred_seq, start_idx

def cal_metrics(actual_seq, pred_seq, start_idx, failure_threshold=0.7):
    # RE
    actual_failure_idx = get_failure_idx(actual_seq, failure_threshold)
    pred_failure_idx = get_failure_idx(pred_seq, failure_threshold) + start_idx
    re = abs(actual_failure_idx - pred_failure_idx) / (actual_failure_idx + 1)
    # RMSE
    rmse = np.sqrt(np.mean((actual_seq[start_idx:] - pred_seq) ** 2))    
    # MAE
    mae = np.mean(np.abs(actual_seq[start_idx:] - pred_seq))
    return re, rmse, mae

# 绘制曲线
def plot(model_names, actual_seq, pred_seqs, start_idx, failure_threshold=0.7, test_battery_name='CS2_35'):
    colors = ['blue', 'green', 'orange', 'purple']
    plt.figure(figsize=(12, 6))
    for i in range(len(pred_seqs)):
        plt.plot(range(start_idx, len(pred_seqs[i]) + start_idx), pred_seqs[i], color=colors[i], linestyle='--', linewidth=2, label=model_names[i])  # 预测值曲线
    plt.plot(actual_seq, color='darkblue', linestyle='-', linewidth=1.5, label='Actual SOH')  # 真实值曲线
    plt.axhline(y=failure_threshold, color='red', linestyle='--', linewidth=2.5, label='Failure Threshold')  # 失效阈值线
    plt.axvline(x=start_idx, color='gray', linestyle='--', linewidth=1, label='Prediction Start Point')  # 预测起始点

    plt.xlabel('Cycles')
    plt.ylabel('SOH')
    plt.legend()
    plt.title(f'SOH degredation of {test_battery_name}')
    plt.show()

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