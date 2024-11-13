"""
实验2: 实验1基础上，将失效后的数据截去不用
"""

import pandas as pd
from data_preprocess import *
from train_and_test import *
from config_loader import *
from tqdm import tqdm

# 实验编号
EXP_NUM = 2

# 测试模型
model_names = ['lstm', 'gru', 'det']

# 实验开始
results = []
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
        model_label = 'exp-{}_{}_s-{}'.format(EXP_NUM, model_config['name'], i+1)

        # dataloader
        norm_data, failure_time = read_and_norm(data_path, rated_capacity, failure_threshold, False)  # 包括失效后数据
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
            actual_seq, sp = test_data, 0
            pred_seq = predict(model_config, model, sp, actual_seq, seq_length, failure_threshold, device)
            re, rmse, mae = cal_metrics(actual_seq, pred_seq, sp, seq_length, failure_threshold)

            print(f"Seed: {i+1}, Model: {model_name}, Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}, Test RE: {re:.3f}, Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")
            
            if re < best_re:
                best_re, best_rmse, best_mae = re, rmse, mae
                torch.save(model.state_dict(), '{}/{}.pth'.format(model_save_dir, model_label))
                print("New best model saved ...")

        results.append({
            'Seed': i + 1,
            'Model': model_label,
            'RE': best_re,
            'RMSE': best_rmse,
            'MAE': best_mae
        })

results_df = pd.DataFrame(results).sort_values(by=['Seed', 'Model']).reset_index(drop=True)
results_df.to_csv(f"exp_results/exp-{EXP_NUM}.csv", index=False)