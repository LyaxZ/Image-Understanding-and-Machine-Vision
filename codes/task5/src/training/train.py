# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import AUGMENTED_DIR, TRAIN_CSV, VAL_CSV, MODELS_DIR, LOGS_DIR
from utils.dataset import CarBrandDataset
from models.improved_cnn import ImprovedCNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
NUM_CLASSES = 6  # 可根据实际类别数修改

def get_transforms():
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    eval_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_trans, eval_trans

def train():
    train_trans, eval_trans = get_transforms()
    train_set = CarBrandDataset(TRAIN_CSV, AUGMENTED_DIR, train_trans)
    val_set = CarBrandDataset(VAL_CSV, AUGMENTED_DIR, eval_trans)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 1. 先实例化模型结构
    model = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    # 2. 加载预训练权重 (假设保存的是 state_dict)
    model_path = os.path.join(MODELS_DIR, f"improved_cnn_{NUM_CLASSES}class_best.pth")
    print(f"Looking for pretrained weights at: {model_path}")
    if os.path.exists(model_path):
        print(f"Loading pretrained weights from {model_path}")
        # 加载 state_dict 并载入模型
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("No pretrained weights found, training from scratch.")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        val_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"improved_cnn_{NUM_CLASSES}class_best.pth"))
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

    # 绘制曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(LOGS_DIR, "training_curves.png"))
    plt.show()
    print(f"Training finished. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()