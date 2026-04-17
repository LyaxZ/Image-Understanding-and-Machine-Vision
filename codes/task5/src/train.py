# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import CarBrandDataset
from improved_cnn import ImprovedCNN

# ========== 加速与兼容设置 ==========
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==================================

# ========== 配置 ==========
BASE_DIR = r"D:\DeepLearning\Image-Understanding-and-Machine-Vision\codes\task5"
DATA_DIR = os.path.join(BASE_DIR, "data", "augmented")
TRAIN_CSV = os.path.join(BASE_DIR, "data", "train.csv")
VAL_CSV   = os.path.join(BASE_DIR, "data", "val.csv")

BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
# ==========================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform

def train():
    print(f"Using device: {DEVICE}")

    train_trans, eval_trans = get_transforms()
    train_dataset = CarBrandDataset(TRAIN_CSV, DATA_DIR, train_trans)
    val_dataset   = CarBrandDataset(VAL_CSV, DATA_DIR, eval_trans)

    # Windows 下 num_workers=0 最稳定
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    model = ImprovedCNN(num_classes=6).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 混合精度（新版 API）
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0.0
    train_losses, val_accs = [], []

    for epoch in range(EPOCHS):
        # ----- 训练 -----
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ----- 验证 -----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(BASE_DIR, "models", "improved_cnn_best.pth"))

        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Best: {best_val_acc:.4f}")

    print(f"\nTraining finished. Best validation accuracy: {best_val_acc:.4f}")

    # 绘制曲线
    os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.subplot(1, 2, 2)
    plt.plot(val_accs)
    plt.title("Validation Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "logs", "improved_cnn_curves.png"))
    plt.show()

if __name__ == "__main__":
    train()