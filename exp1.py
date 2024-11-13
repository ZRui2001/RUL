"""
实验1: RE作为模型参数保存的标准
"""

from data_preprocess import *
from train_and_test import *
import yaml
import random
from tqdm import tqdm

# 实验编号
EXP_NUM = 1

# 测试模型
model_names = ['LSTM', 'GRU', 'DeTransformer']

# 获取配置参数
with open('config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)
data_config, models_config, start_points, model_save_dir, seeds = (
    config['data']['CALCE'],       # 仅使用 CALCE
    config['models'],
    config['start_points'],        # [0.3, 0.5, 0.7]
    config['model_save_dir'],
    config['seeds']
)
data_path, test_bat, seq_length, rated_capacity, failure_threshold = (
    data_config['data_path'],
    data_config['test_bat'],
    data_config['seq_length'],
    data_config['rated_capacity'],
    data_config['failure_threshold']
)
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

# 实验开始
results = {}
for i in tqdm(range(len(seeds))):
    # 设置随机种子
    set_seed(seeds[i])

    for model_name in tqdm(model_names):
        # 参数
        model_config = models_config[model_name]
        batch_size, lr, optim_name, epochs, metric, alpha = (
            model_config['batch_size'],
            model_config['lr'],
            model_config['optimizer'],
            model_config['epochs'],
            model_config['metric'],
            model_config.get('alpha', None)
        )

        # dataloader
        norm_data, failure_time = read_and_norm(data_path, rated_capacity, failure_threshold)  # 包括失效后数据
        train_data, test_data = split_data(norm_data, test_bat)
        train_loader, test_loader = load_data(train_data, test_data, seq_length, batch_size)

        # 模型、优化器
        model = get_model(model_config, device)
        optimizer = get_optimizer(optim_name, model, lr, alpha)
        criterion = nn.MSELoss()

        # 训练，保存测试集上 re 最好的模型
        best_re, best_rmse, best_mae = float('inf'), float('inf'), float('inf')
        for epoch in range(epochs):
            train_loss = train_epoch(model_config, model, train_loader, device, optimizer, criterion)
            test_loss = test_epoch(model_config, model, test_loader, device, criterion)
            # 从第一个窗口开始迭代预测，得到预测曲线（包含第一个窗口）
            actual_seq, start_idx = test_data, seq_length
            pred_seq = predict(model_config, model, start_idx, actual_seq, seq_length, device)
            re, rmse, mae = cal_metrics(actual_seq, pred_seq, start_idx, failure_threshold)

            print(f"Seed: {i+1}, Model: {model_name}, Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test RE: {re:.3f}, Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
            
            if re < best_re:
                best_re, best_rmse, best_mae = re, rmse, mae
                torch.save(model.state_dict(), '{}/exp-{}_s-{}_{}.pth'.format(model_save_dir, EXP_NUM, i+1, model_config['name']))
                print("New best model saved ...")
        results[(i + 1, model_name)] = {
            're': best_re,
            'rmse': best_rmse,
            'mae': best_mae
        }

# 逐行打印每个 (seed, model_name) 组合的最佳结果
for (seed, model_name), result in sorted(results.items(), key=lambda x: (x[0][1], x[0][0])):
    print(f"Seed: {seed}, Model: {model_name}, Best RE: {result['re']:.3f}, Best RMSE: {result['rmse']:.4f}, Best MAE: {result['mae']:.4f}")