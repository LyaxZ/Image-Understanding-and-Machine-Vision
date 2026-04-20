# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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
from models.efficientnet import EfficientNetTransfer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
BATCH_SIZE = 64
EPOCHS = 8
LR = 1e-3

# 配置区添加模型选择
MODEL_NAME = "efficientnet"   # 可选 "improved" 或 "efficientnet"
NUM_CLASSES = 6              # 根据你的任务修改
PRETRAINED = True             # EfficientNet 建议使用预训练权重



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

def load_pretrained(model, model_name, num_classes, device):
    """
    根据模型名称加载对应的预训练权重
    - ImprovedCNN: 尝试加载 task5/models/improved_cnn_{num_classes}class_best.pth
                    若类别数不匹配，则只加载卷积层（迁移学习）
    - EfficientNet: 使用 timm 自带的 ImageNet 预训练，无需额外文件
    """
    if model_name == "improved":
        model_path = os.path.join(MODELS_DIR, f"improved_cnn_{num_classes}class_best.pth")
        print(f"Looking for ImprovedCNN pretrained weights at: {model_path}")
        if os.path.exists(model_path):
            print(f"Loading pretrained weights from {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model_dict = model.state_dict()
            # 检查类别数是否匹配（通过 fc2 层的输出维度判断）
            pretrained_num_classes = state_dict['fc2.weight'].shape[0]
            if pretrained_num_classes != num_classes:
                print(f"⚠️ 类别数不匹配 (预训练: {pretrained_num_classes} → 当前: {num_classes})，仅加载卷积层")
                # 过滤掉全连接层参数
                filtered_dict = {}
                for k, v in state_dict.items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        if not k.startswith('fc'):   # 排除分类头
                            filtered_dict[k] = v
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                print(f"✅ 成功加载 {len(filtered_dict)} 个卷积层参数")
            else:
                model.load_state_dict(state_dict)
                print("✅ 完整加载预训练权重")
        else:
            print("No ImprovedCNN pretrained weights found, training from scratch.")

    elif model_name == "efficientnet":
        # EfficientNet 已经在初始化时通过 timm 的 pretrained=True 加载了 ImageNet 权重
        print("✅ EfficientNet using ImageNet pretrained weights (built-in).")

    else:
        print(f"⚠️ Unknown model name '{model_name}', no pretrained weights loaded.")

    return model

def train():
    train_trans, eval_trans = get_transforms()
    train_set = CarBrandDataset(TRAIN_CSV, AUGMENTED_DIR, train_trans)
    val_set = CarBrandDataset(VAL_CSV, AUGMENTED_DIR, eval_trans)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 模型初始化部分
    if MODEL_NAME == "improved":
        model = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "efficientnet":
        model = EfficientNetTransfer(num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(DEVICE)
    else:
        raise ValueError(f"未知模型: {MODEL_NAME}")
    
    model = load_pretrained(model, MODEL_NAME, NUM_CLASSES, DEVICE)
    
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
            save_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}_{NUM_CLASSES}class_best.pth")
            torch.save(model.state_dict(), save_path)
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
    plt.savefig(os.path.join(LOGS_DIR, f"training_curves_{NUM_CLASSES}class.png"))
    plt.show()
    print(f"Training finished. Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    train()