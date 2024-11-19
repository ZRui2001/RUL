"""
实验0：测试集样本mse最低时保存参数；使用全量数据训练
"""

import pandas as pd
from data_preprocess import *
from train_and_test import *
from config_loader import *
from tqdm import tqdm

# 实验编号
EXP_NUM = 0

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
        batteries_df = read_and_norm(data_path, rated_capacity, failure_threshold)  
        train_df, val_df, test_df, train_loader, val_loader, test_loader = load_data(batteries_df, val_bat, test_bat, seq_length, batch_size)  # 包括失效后数据

        # 模型、优化器
        model = get_model(model_config, device)
        optimizer = get_optimizer(optim_name, model, lr, alpha)
        criterion = nn.MSELoss()

        # 训练，保存验证集上 re 最好的模型
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = train_epoch(model_config, model, train_loader, device, optimizer, criterion)
            val_loss = test_epoch(model_config, model, test_loader, device, criterion)
            # 从第一个窗口开始迭代预测，得到预测曲线（包含第一个窗口）
            val_seq, sp = val_df['capacity'].to_numpy(), 0.0
            pred_seq = predict(model_config, model, sp, val_seq, seq_length, failure_threshold, device)
            re, rmse, mae = cal_metrics(val_seq, pred_seq, sp, seq_length, failure_threshold)

            print(f"Seed: {i+1}, Model: {model_name}, Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}, Val RE: {re:.3f}, Val RMSE: {rmse:.4f}, Val MAE: {mae:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
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