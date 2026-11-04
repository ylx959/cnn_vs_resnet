import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from cnn import BasicCNN, PlainCNN
from data import ROOT, get_dataloaders
from resnet import ResNet20

MODEL_DIR = ROOT / 'model'
RUNS_DIR = ROOT / 'results' / 'runs'

# BasicCNN 沒有 BatchNorm，學習率 0.1 容易發散，所以用較小的值
LEARNING_RATES = {'cnn': 0.01, 'plain': 0.1, 'resnet': 0.1}


def build_model(name):
    if name == 'cnn':
        return BasicCNN()
    if name == 'plain':
        return PlainCNN()
    return ResNet20()


def get_device():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def synchronize(device):
    # GPU 運算是非同步的，計時前後要等它真正做完
    if device.type == 'mps':
        torch.mps.synchronize()


def train_and_evaluate(model, trainloader, valloader, epochs, lr, device, save_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],
               'val_acc': [], 'seconds': []}
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        synchronize(device)
        start_time = time.perf_counter()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向傳播
            outputs = model(inputs)
            # 計算 loss
            loss = criterion(outputs, labels)
            # 反向傳播
            loss.backward()
            # 更新權重
            optimizer.step()

            running_loss += loss.item()
            total += labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()

        synchronize(device)
        seconds = time.perf_counter() - start_time
        train_loss = running_loss / len(trainloader)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                val_running_loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_running_loss / len(valloader)
        val_acc = 100 * correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['seconds'].append(seconds)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch+1}/{epochs}] - LR: {current_lr:.6f} - Train Loss: {train_loss:.4f} - "
              f"Train Acc: {train_acc:.2f}% - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% - {seconds:.1f}s",
              flush=True)

        scheduler.step()

        # 儲存驗證集準確率最高的模型（測試集留到 evalute.py 才使用）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['cnn', 'plain', 'resnet'], required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(args.seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_name = f"{args.model}_seed{args.seed}"

    trainloader, valloader, _ = get_dataloaders(batch_size=128)
    model = build_model(args.model).to(device)

    print(f"\n--- Training {run_name} ---")
    history = train_and_evaluate(
        model, trainloader, valloader, epochs=args.epochs, lr=LEARNING_RATES[args.model],
        device=device, save_path=MODEL_DIR / f"{run_name}.pth")

    with open(RUNS_DIR / f"{run_name}.json", 'w') as f:
        json.dump(history, f, indent=2)
