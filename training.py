import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from models import AttentionConvModel, SimpleAttentionConvModel
from utils import *  # 包含评价指标函数 mse, pearson, ci

# -------------------
# 损失函数
# -------------------
loss_nll = nn.GaussianNLLLoss(full=True)


def collate_fn(batch):
    """自定义批处理函数，处理字典格式的数据"""
    compound_sequences = torch.stack([item['compound_sequence'] for item in batch])
    protein_embeddings = torch.stack([item['protein_embedding'] for item in batch])
    affinities = torch.stack([item['affinity'] for item in batch])

    return {
        'compound_sequence': compound_sequences,
        'protein_embedding': protein_embeddings,
        'affinity': affinities
    }


def train(model, device, train_loader, optimizer, scheduler, epoch, grad_accum_steps=1):
    """训练函数 - 使用NLL损失，返回MSE、NLL和CI"""
    model.train()
    total_nll_loss = 0
    total_mse_loss = 0
    total_preds, total_labels = torch.empty(0), torch.empty(0)

    optimizer.zero_grad(set_to_none=True)
    for batch_idx, data in enumerate(train_loader):
        # 将数据移动到设备
        compound_seq = data['compound_sequence'].to(device, non_blocking=True)
        protein_embed = data['protein_embedding'].to(device, non_blocking=True)
        target = data['affinity'].to(device, non_blocking=True)

        batch_data = {
            'compound_sequence': compound_seq,
            'protein_embedding': protein_embed,
            'affinity': target
        }

        # 使用新 API 的 autocast
        with torch.amp.autocast(device_type=device.type):
            # 模型需要返回均值和方差
            mean, var = model(batch_data)
            nll_loss = loss_nll(mean, target, var)
            mse_loss = F.mse_loss(mean, target)

        # 使用全局 scaler
        scaler.scale(nll_loss).backward()

        # 梯度累积
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        total_nll_loss += nll_loss.item()
        total_mse_loss += mse_loss.item()

        total_preds = torch.cat((total_preds, mean.detach().cpu()), 0)
        total_labels = torch.cat((total_labels, target.detach().cpu()), 0)

    labels = total_labels.numpy().flatten()
    preds = total_preds.numpy().flatten()

    # 计算所有指标
    avg_nll_loss = total_nll_loss / len(train_loader)
    avg_mse_loss = total_mse_loss / len(train_loader)
    ci_val = ci(labels, preds)

    return avg_nll_loss, avg_mse_loss, ci_val


def evaluating(model, device, loader):
    """评估函数 - 使用NLL损失，返回MSE、NLL和CI"""
    model.eval()
    total_nll_loss = 0
    total_mse_loss = 0
    total_preds, total_labels = torch.empty(0), torch.empty(0)

    with torch.no_grad():
        for data in loader:
            # 将数据移动到设备
            compound_seq = data['compound_sequence'].to(device, non_blocking=True)
            protein_embed = data['protein_embedding'].to(device, non_blocking=True)
            target = data['affinity'].to(device, non_blocking=True)

            batch_data = {
                'compound_sequence': compound_seq,
                'protein_embedding': protein_embed,
                'affinity': target
            }

            # 使用新 API 的 autocast
            with torch.amp.autocast(device_type=device.type):
                mean, var = model(batch_data)
                nll_loss = loss_nll(mean, target, var)
                mse_loss = F.mse_loss(mean, target)

            total_nll_loss += nll_loss.item()
            total_mse_loss += mse_loss.item()

            total_preds = torch.cat((total_preds, mean.detach().cpu()), 0)
            total_labels = torch.cat((total_labels, target.detach().cpu()), 0)

    labels = total_labels.numpy().flatten()
    preds = total_preds.numpy().flatten()

    # 计算所有指标
    avg_nll_loss = total_nll_loss / len(loader)
    avg_mse_loss = total_mse_loss / len(loader)
    ci_val = ci(labels, preds)

    return avg_nll_loss, avg_mse_loss, ci_val


if len(sys.argv) < 3:
    print("Usage: python training.py <dataset_index> <model_index> [cuda_index]")
    print("Models: 0 for AttentionConvModel, 1 for SimpleAttentionConvModel")
    sys.exit(1)

datasets = ['davis', 'kiba']
dataset = datasets[int(sys.argv[1])]
modeling = [AttentionConvModel, SimpleAttentionConvModel][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[3]))
print('Using device:', cuda_name)

best_ci = -1
best_epoch = -1

# -------------------
# 超参设置
# -------------------
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
LR = 0.001
NUM_EPOCHS = 200
GRAD_ACCUM_STEPS = 1

print('\nRunning on', model_st + '_' + dataset)
print('Using NLL Loss')
print('Evaluation metrics: MSE, NLL, CI')
print('Model selection metric: CI (higher is better)')

# 使用新的序列数据文件
processed_data_file_train = f'data/processed/{dataset}_train_sequence.pt'
processed_data_file_test = f'data/processed/{dataset}_test_sequence.pt'

if not (os.path.isfile(processed_data_file_train) and os.path.isfile(processed_data_file_test)):
    print('Please run create_data.py first to generate sequence data!')
    sys.exit(1)
else:
    # 加载序列数据
    train_data = torch.load(processed_data_file_train)
    test_data = torch.load(processed_data_file_test)

    # 创建数据加载器
    train_loader = DataLoader(
        train_data,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    # 初始化新模型
    if modeling == AttentionConvModel:
        model = modeling(n_output=1, dropout=0.3).to(device)
    else:  # SimpleAttentionConvModel
        model = modeling(n_output=1, dropout=0.3).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 学习率调度器
    steps_per_epoch = max(1, len(train_loader) // max(1, GRAD_ACCUM_STEPS))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=NUM_EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos"
    )

    # 在确认 device 后再初始化 scaler
    scaler = torch.amp.GradScaler()

    model_file_name = f'model_{model_st}_{dataset}_nll.model'

    results = {
        "train": [],
        "test": [],
        "config": {
            "dataset": dataset,
            "model": model_st,
            "loss": "NLL",
            "metrics": ["MSE", "NLL", "CI"],
            "selection_metric": "CI"
        }
    }

    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(NUM_EPOCHS):
        train_nll, train_mse, train_ci = train(model, device, train_loader, optimizer, scheduler, epoch + 1,
                                               GRAD_ACCUM_STEPS)
        test_nll, test_mse, test_ci = evaluating(model, device, test_loader)

        results["train"].append({
            "epoch": epoch + 1,
            "nll_loss": train_nll,
            "mse_loss": train_mse,
            "ci": train_ci
        })
        results["test"].append({
            "epoch": epoch + 1,
            "nll_loss": test_nll,
            "mse_loss": test_mse,
            "ci": test_ci
        })

        print(f"Epoch: {epoch + 1}")
        print(f"Train -> NLL: {train_nll:.4f}, MSE: {train_mse:.4f}, CI: {train_ci:.4f}")
        print(f"Test  -> NLL: {test_nll:.4f}, MSE: {test_mse:.4f}, CI: {test_ci:.4f}")
        print("---")

        if test_ci > best_ci:
            torch.save(model.state_dict(), model_file_name)
            best_epoch = epoch + 1
            best_ci = test_ci
            print(f'Test CI improved at epoch {best_epoch}, Best CI: {best_ci:.4f}')
        else:
            print(f'No improvement. Best CI: {best_ci:.4f} from epoch {best_epoch}')

    json_file_name = f'training_results_{model_st}_{dataset}_nll.json'
    with open(json_file_name, "w") as f:
        json.dump(results, f, indent=4)
    print(f'Results saved to {json_file_name}')

    # 加载最佳模型进行最终测试
    print("\n=== Final Evaluation with Best Model ===")
    model.load_state_dict(torch.load(model_file_name))
    final_test_nll, final_test_mse, final_test_ci = evaluating(model, device, test_loader)
    print(f"Final Test Results -> NLL: {final_test_nll:.4f}, MSE: {final_test_mse:.4f}, CI: {final_test_ci:.4f}")