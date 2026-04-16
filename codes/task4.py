import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import numpy as np
import random
import time
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    HAS_ADV_PLOT = True
except ImportError:
    HAS_ADV_PLOT = False
    print("提示：安装 scikit-learn 和 seaborn 可解锁混淆矩阵等高级可视化图表！")

# ==================== 全局配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = r".\outputs\cifar10_comparison"

# ==================== 数据导入函数 ====================

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return train_transform, test_transform

def load_cifar10_data(data_root='./Data/inputs', batch_size=64, train_samples=None, test_samples=None):
    """
    加载 CIFAR-10 数据集，支持随机抽样
    Args:
        train_samples: 训练集抽样数量，None表示使用全部(50000)
        test_samples: 测试集抽样数量，None表示使用全部(10000)
    """
    print("="*70)
    print("      加载 CIFAR-10 数据集")
    print("="*70)
    
    train_transform, test_transform = get_transforms()
    
    full_train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    full_test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)
    
    # 数据抽样逻辑
    if train_samples and train_samples < len(full_train_dataset):
        indices = random.sample(range(len(full_train_dataset)), train_samples)
        train_dataset = Subset(full_train_dataset, indices)
        print(f"[OK] 已随机抽取 {train_samples} 张训练集样本")
    else:
        train_dataset = full_train_dataset
        
    if test_samples and test_samples < len(full_test_dataset):
        indices = random.sample(range(len(full_test_dataset)), test_samples)
        test_dataset = Subset(full_test_dataset, indices)
        print(f"[OK] 已随机抽取 {test_samples} 张测试集样本")
    else:
        test_dataset = full_test_dataset
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    class_names = full_train_dataset.classes
    
    print(f"\n[OK] 实际训练批次数：{len(train_loader)}")
    print(f"[OK] 实际测试批次数：{len(test_loader)}")
    print(f"[OK] 类别：{class_names}")
    
    return train_loader, test_loader, class_names

# ==================== 模型定义函数 ====================

# 五层卷积网络
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.pool(self.relu(self.bn5(self.conv5(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)

# 2. 轻量级深度可分离卷积网络
class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DepthwiseSeparableCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dw_conv1 = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.pw_conv1 = nn.Conv2d(32, 64, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dw_conv2 = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pw_conv2 = nn.Conv2d(64, 128, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dw_conv3 = nn.Conv2d(128, 128, 3, padding=1, groups=128)
        self.pw_conv3 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.pw_conv1(self.relu(self.dw_conv1(x))))))
        x = self.pool(self.relu(self.bn3(self.pw_conv2(self.relu(self.dw_conv2(x))))))
        x = self.pool(self.relu(self.bn4(self.pw_conv3(self.relu(self.dw_conv3(x))))))
        x = self.global_avg_pool(x)
        return self.fc(x.view(x.size(0), -1))

# 3. 微型残差网络
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class MiniResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MiniResNet, self).__init__()
        self.initial = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.layer1 = nn.Sequential(*[ResidualBlock(32) for _ in range(2)])
        self.trans1 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.layer2 = nn.Sequential(*[ResidualBlock(64) for _ in range(2)])
        self.trans2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.trans1(x)
        x = self.layer2(x)
        x = self.trans2(x)
        return self.fc(self.global_avg_pool(x).view(x.size(0), -1))

def create_model(model_type='cnn'):
    models = {
        'cnn': CNNClassifier,
        'dw_cnn': DepthwiseSeparableCNN,
        'resnet': MiniResNet
    }
    if model_type not in models: raise ValueError(f"不支持的模型: {model_type}")
    
    model = models[model_type]().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[OK] 创建模型：[{model_type.upper()}] 参数量: {total_params:,}")
    return model

# ==================== 训练与测试函数 ====================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch, model_name):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(train_loader, desc=f'[{model_name}] Epoch {epoch}')
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs); loss = criterion(outputs, targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1); total += targets.size(0); correct += predicted.eq(targets).sum().item()
        progress_bar.set_postfix({'loss': f'{running_loss / (progress_bar.n + 1):.4f}', 'acc': f'{100. * correct / total:.2f}%'})
    return running_loss / len(train_loader), 100. * correct / total

@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval(); correct, total = 0, 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        _, predicted = model(inputs).max(1); total += targets.size(0); correct += predicted.eq(targets).sum().item()
    return 100. * correct / total

def test_model_detailed(model, test_loader, class_names):
    """返回总体准确率、各类别准确率字典、所有真实标签、所有预测标签"""
    model.eval(); correct, total = 0, 0
    class_correct = np.zeros(len(class_names)); class_total = np.zeros(len(class_names))
    all_preds, all_labels = [], []
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs); _, predicted = outputs.max(1)
        total += targets.size(0); correct += predicted.eq(targets).sum().item()
        for i in range(targets.size(0)):
            label, pred = targets[i].item(), predicted[i].item()
            class_total[label] += 1
            if pred == label: class_correct[label] += 1
            all_preds.append(pred); all_labels.append(label)
    
    class_accs = {name: 100. * class_correct[i] / class_total[i] for i, name in enumerate(class_names) if class_total[i] > 0}
    return 100. * correct / total, class_accs, np.array(all_labels), np.array(all_preds)

def save_checkpoint(model, optimizer, epoch, accuracy, model_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f'best_{model_name}.pth')
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'accuracy': accuracy}, path)

# ==================== 增强版可视化系统 ====================

def set_plot_style():
    if not HAS_ADV_PLOT: return
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

def plot_comparison_curves(all_histories, save_dir):
    """绘制多模型 Loss 和 Acc 对比曲线"""
    if not HAS_ADV_PLOT: return
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['blue', 'orange', 'green']
    
    for idx, (name, history) in enumerate(all_histories.items()):
        ax1.plot(history['train_loss'], color=colors[idx], linestyle='-', label=f'{name} 训练Loss')
        ax1.plot(history['test_loss'], color=colors[idx], linestyle='--', label=f'{name} 验证Loss')
        ax2.plot(history['train_acc'], color=colors[idx], linestyle='-', label=f'{name} 训练Acc')
        ax2.plot(history['test_acc'], color=colors[idx], linestyle='--', label=f'{name} 验证Acc')

    ax1.set_title('损失曲线对比'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.set_title('准确率曲线对比'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'curves_comparison.png'), dpi=150)
    plt.close()

def plot_time_and_params_comparison(stats, save_dir):
    """绘制训练时间与参数量对比"""
    if not HAS_ADV_PLOT: return
    set_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    names = list(stats.keys())
    times = [stats[n]['time'] for n in names]
    params = [stats[n]['params'] for n in names]
    
    ax1.bar(names, times, color=['#4C72B0', '#DD8452', '#55A868'])
    ax1.set_title('总训练耗时对比 (秒)'); ax1.set_ylabel('Seconds')
    for i, v in enumerate(times): ax1.text(i, v + 0.5, f"{v:.1f}s", ha='center')
    
    ax2.bar(names, params, color=['#4C72B0', '#DD8452', '#55A868'])
    ax2.set_title('模型参数量对比'); ax2.set_ylabel('Parameters')
    for i, v in enumerate(params): ax2.text(i, v + max(params)*0.01, f"{v/1000:.1f}K", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_params_comparison.png'), dpi=150)
    plt.close()

def plot_class_accuracy_comparison(all_class_accs, save_dir):
    """绘制各类别准确率分组柱状图"""
    if not HAS_ADV_PLOT: return
    set_plot_style()
    class_names = list(list(all_class_accs.values())[0].keys())
    x = np.arange(len(class_names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for idx, (name, accs) in enumerate(all_class_accs.items()):
        values = [accs[c] for c in class_names]
        ax.bar(x - width + idx*width, values, width, label=name)

    ax.set_ylabel('Accuracy (%)'); ax.set_title('各模型在10个类别上的准确率对比'); ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=15)
    ax.legend(); ax.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_acc_comparison.png'), dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir):
    """绘制单个模型的混淆矩阵"""
    if not HAS_ADV_PLOT: return
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - 混淆矩阵'); plt.ylabel('真实标签'); plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{model_name}.png'), dpi=150)
    plt.close()

# ==================== 主程序 ====================

def main():
    print("="*70)
    print("      CIFAR-10 图像分类 - 多模型全维对比实验")
    print("="*70)
    
    # ==================== 核心配置 ====================
    DATA_ROOT = r'.\Data\inputs'
    BATCH_SIZE = 150
    EPOCHS = 15
    LEARNING_RATE = 0.005
    
    # 🌟 数据集抽样配置（设为 None 则使用全部数据，小数据测试建议设为 5000 和 1000）
    TRAIN_SAMPLES = 25000 
    TEST_SAMPLES = 5000   
    
    # 🌟 参与对比的模型列表
    MODELS_TO_TRAIN = ['cnn', 'dw_cnn', 'resnet']
    MODEL_DISPLAY_NAMES = {'cnn': 'VGG式CNN', 'dw_cnn': '轻量级DW-CNN', 'resnet': '微型ResNet'}
    
    # ==================== 加载数据 ====================
    train_loader, test_loader, class_names = load_cifar10_data(
        data_root=DATA_ROOT, batch_size=BATCH_SIZE,
        train_samples=TRAIN_SAMPLES, test_samples=TEST_SAMPLES
    )
    
    all_histories = {}
    all_class_accs = {}
    model_stats = {}
    all_true_labels = None
    
    # ==================== 循环训练与评估 ====================
    for model_type in MODELS_TO_TRAIN:
        display_name = MODEL_DISPLAY_NAMES[model_type]
        print(f"\n{'='*70}\n>>> 开始训练模型：{display_name} <<<\n{'='*70}")
        
        model = create_model(model_type)
        model_stats[model_type] = {'params': sum(p.numel() for p in model.parameters())}
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
        best_acc = 0.0
        start_time = time.time()
        
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch, display_name)
            scheduler.step()
            
            # 计算验证Loss
            model.eval(); val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs); val_loss += criterion(outputs, targets).item()
            val_loss /= len(test_loader)
            
            test_acc = evaluate_model(model, test_loader)
            
            history['train_loss'].append(train_loss); history['test_loss'].append(val_loss)
            history['train_acc'].append(train_acc); history['test_acc'].append(test_acc)
            
            if test_acc > best_acc:
                best_acc = test_acc
                save_checkpoint(model, optimizer, epoch, test_acc, model_type)
                
        train_time = time.time() - start_time
        model_stats[model_type]['time'] = train_time
        all_histories[display_name] = history
        print(f"\n[OK] {display_name} 训练完成！最佳验证准确率: {best_acc:.2f}%, 耗时: {train_time:.1f}秒")
        
        # 获取详细测试结果用于画图
        overall_acc, class_accs, true_labels, pred_labels = test_model_detailed(model, test_loader, class_names)
        all_class_accs[display_name] = class_accs
        plot_confusion_matrix(true_labels, pred_labels, class_names, display_name, OUTPUT_DIR)
        
        if all_true_labels is None: all_true_labels = true_labels # 只要存一份真实标签就行
        
        # 释放显存
        del model; torch.cuda.empty_cache()

    # ==================== 绘制综合对比图 ====================
    print("\n" + "="*70)
    print("生成对比可视化图表...")
    print("="*70)
    
    plot_comparison_curves(all_histories, OUTPUT_DIR)
    plot_time_and_params_comparison(model_stats, OUTPUT_DIR)
    plot_class_accuracy_comparison(all_class_accs, OUTPUT_DIR)
    
    print(f"\n[OK] 所有实验完成！图表已保存至：{OUTPUT_DIR}")
    print("  - curves_comparison.png : 损失与准确率曲线对比")
    print("  - time_params_comparison.png : 训练速度与参数量对比")
    print("  - class_acc_comparison.png : 十大类别分类能力对比")
    print("  - confusion_matrix_*.png : 各模型混淆矩阵热力图")

if __name__ == "__main__":
    main()
