# -*- coding: utf-8 -*-
"""
完整可视化：准确率对比柱状图 + 混淆矩阵 + Grad-CAM
支持 ImprovedCNN 和 EfficientNet
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# 中文字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AUGMENTED_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, MODELS_DIR, LOGS_DIR
from utils.dataset import CarBrandDataset
from models.improved_cnn import ImprovedCNN
from models.efficientnet import EfficientNetTransfer

# ========== 配置（请根据需要修改） ==========
MODEL_NAME = "efficientnet"          # 可选 "improved" 或 "efficientnet"
NUM_CLASSES = 6                 # 6 或 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 根据模型名自动生成权重文件路径
if MODEL_NAME == "improved":
    MODEL_PATH = os.path.join(MODELS_DIR, f"improved_cnn_{NUM_CLASSES}class_best.pth")
elif MODEL_NAME == "efficientnet":
    MODEL_PATH = os.path.join(MODELS_DIR, f"efficientnet_{NUM_CLASSES}class_best.pth")
else:
    raise ValueError(f"Unsupported MODEL_NAME: {MODEL_NAME}")

CLASS_NAMES = sorted([d for d in os.listdir(AUGMENTED_DIR) if os.path.isdir(os.path.join(AUGMENTED_DIR, d))])
assert len(CLASS_NAMES) == NUM_CLASSES, f"类别数量不匹配，应为 {len(CLASS_NAMES)}"

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ========== 加载模型 ==========
def load_model():
    if MODEL_NAME == "improved":
        model = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)
    elif MODEL_NAME == "efficientnet":
        model = EfficientNetTransfer(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {MODEL_NAME}")

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"✅ 模型加载成功：{MODEL_PATH}")
    else:
        raise FileNotFoundError(f"模型权重文件不存在：{MODEL_PATH}")
    model.eval()
    return model

# ========== 评估函数 ==========
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total

# ========== 1. 准确率对比柱状图 ==========
def plot_accuracy_comparison(train_acc, val_acc, test_acc, save_path):
    categories = ['训练集', '验证集', '测试集']
    accuracies = [train_acc, val_acc, test_acc]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.8)
    plt.ylim(0, 1.0)
    plt.ylabel('准确率')
    plt.title(f'{MODEL_NAME} 准确率对比 ({NUM_CLASSES} 分类)')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{acc:.4f}', ha='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 准确率对比图已保存：{save_path}")

# ========== 2. 混淆矩阵 ==========
def plot_confusion_matrix(model, dataloader, save_path):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    print("\n📊 分类报告：")
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(f'{MODEL_NAME} 混淆矩阵 ({NUM_CLASSES} 分类)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ 混淆矩阵已保存：{save_path}")

# ========== 3. Grad-CAM 可视化 ==========
def get_target_layer(model):
    """根据模型类型返回适合的 Grad-CAM 目标层"""
    if MODEL_NAME == "improved":
        return model.conv4_2   # ImprovedCNN 最后一个卷积层
    elif MODEL_NAME == "efficientnet":
        # EfficientNet 的最后一个卷积层位于 backbone 的某个位置
        # 通常选择 blocks 的最后一层
        return model.backbone.blocks[-1][-1]   # 可能因版本而异，可打印 model 调整
    else:
        raise ValueError(f"Unknown model for Grad-CAM: {MODEL_NAME}")

def plot_gradcam_examples(model, dataset, save_dir, num_per_class=1):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("⚠️ grad-cam 未安装，跳过 Grad-CAM 可视化。安装命令：pip install grad-cam")
        return

    os.makedirs(save_dir, exist_ok=True)
    target_layer = get_target_layer(model)

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    class_samples = {cls: [] for cls in range(NUM_CLASSES)}
    for img_tensor, label in dataset:
        if len(class_samples[label]) < num_per_class:
            class_samples[label].append(img_tensor)
        if all(len(v) >= num_per_class for v in class_samples.values()):
            break

    for cls_idx, tensors in class_samples.items():
        for i, img_tensor in enumerate(tensors):
            img_tensor = img_tensor.to(DEVICE)
            cam = GradCAM(model=model, target_layers=[target_layer])
            grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0))[0, :]

            img_np = inv_normalize(img_tensor).cpu().numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)

            cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

            cls_name = CLASS_NAMES[cls_idx]
            save_path = os.path.join(save_dir, f"{cls_name}_gradcam_{i+1}.png")
            plt.imsave(save_path, cam_image)
            print(f"✅ Grad-CAM 已保存：{save_path}")

# ========== 主函数 ==========
def main():
    model = load_model()
    criterion = nn.CrossEntropyLoss()

    # 数据集
    train_dataset = CarBrandDataset(TRAIN_CSV, AUGMENTED_DIR, transform)
    val_dataset = CarBrandDataset(VAL_CSV, AUGMENTED_DIR, transform)
    test_dataset = CarBrandDataset(TEST_CSV, AUGMENTED_DIR, transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # 计算准确率
    print("正在计算各数据集准确率...")
    train_acc = evaluate(model, train_loader, criterion)[1]
    val_acc = evaluate(model, val_loader, criterion)[1]
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"\n训练集准确率: {train_acc:.4f}")
    print(f"验证集准确率: {val_acc:.4f}")
    print(f"测试集准确率: {test_acc:.4f} | 测试集损失: {test_loss:.4f}")

    # 1. 准确率对比柱状图
    acc_path = os.path.join(LOGS_DIR, f"{MODEL_NAME}_accuracy_comparison_{NUM_CLASSES}class.png")
    plot_accuracy_comparison(train_acc, val_acc, test_acc, acc_path)

    # 2. 混淆矩阵
    cm_path = os.path.join(LOGS_DIR, f"{MODEL_NAME}_confusion_matrix_{NUM_CLASSES}class.png")
    plot_confusion_matrix(model, test_loader, cm_path)

    # 3. Grad-CAM
    gradcam_dir = os.path.join(LOGS_DIR, f"{MODEL_NAME}_gradcam_{NUM_CLASSES}class")
    plot_gradcam_examples(model, test_dataset, gradcam_dir, num_per_class=1)

    print("\n🎉 所有可视化完成！")

if __name__ == "__main__":
    main()