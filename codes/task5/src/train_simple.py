# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CarBrandDataset, get_transforms
from simple_cnn import SimpleCNN

# 配置
BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
DATA_DIR = os.path.join(BASE_DIR, "data", "augmented")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "data", "val.csv")

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Using device: {DEVICE}")

    # 数据加载
    train_trans, eval_trans = get_transforms()
    train_dataset = CarBrandDataset(TRAIN_CSV, DATA_DIR, train_trans)
    val_dataset = CarBrandDataset(VAL_CSV, DATA_DIR, eval_trans)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型、损失函数、优化器
    model = SimpleCNN(num_classes=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        # 训练
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
        scheduler.step()

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accs.append(val_acc)
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Acc = {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), os.path.join(BASE_DIR, "models", "simple_cnn.pth"))
    print("Model saved.")

    # 绘制曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "logs", "simple_cnn_curves.png"))
    plt.show()

if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    train()