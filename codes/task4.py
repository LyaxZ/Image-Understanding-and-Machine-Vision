import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os
import numpy as np
from tqdm import tqdm


# ==================== 全局配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = r".\outputs\cifar10_classification"


# ==================== 数据导入函数 ====================

def get_transforms():
    """
    获取数据预处理变换
    
    Returns:
        train_transform: 训练集变换
        test_transform: 测试集变换
    """
    # 训练集变换：包含增强的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    
    # 测试集变换：仅做基本转换
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return train_transform, test_transform


def load_cifar10_data(data_root='./Data/inputs', batch_size=64):
    """
    加载 CIFAR-10 数据集
    
    Args:
        data_root: 数据集根目录路径
        batch_size: 批次大小
        
    Returns:
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
        class_names: 类别名称列表
    """
    print("="*70)
    print("      加载 CIFAR-10 数据集")
    print("="*70)
    
    # 获取数据变换
    train_transform, test_transform = get_transforms()
    
    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=train_transform
    )
    
    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 类别名称
    class_names = train_dataset.classes
    
    print(f"\n[OK] 训练集样本数：{len(train_dataset)}")
    print(f"[OK] 测试集样本数：{len(test_dataset)}")
    print(f"[OK] 批次大小：{batch_size}")
    print(f"[OK] 训练批次数：{len(train_loader)}")
    print(f"[OK] 测试批次数：{len(test_loader)}")
    print(f"[OK] 类别：{class_names}")
    print(f"[OK] 使用设备：{DEVICE}")
    
    return train_loader, test_loader, class_names


# ==================== 模型定义函数 ====================

class NeuralNetworkClassifier(nn.Module):
    """
    三层神经网络分类器（带 ReLU 激活函数）
    """
    def __init__(self, input_size=32*32*3, hidden_size1=1024, hidden_size2=512, hidden_size3=256, num_classes=10):
        super(NeuralNetworkClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            # 第一层：输入层 -> 隐藏层 1 (1024 节点)
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # 第二层：隐藏层 1 -> 隐藏层 2 (512 节点)
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # 第三层：隐藏层 2 -> 隐藏层 3 (256 节点)
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第四层：隐藏层 3 -> 输出层
            nn.Linear(hidden_size3, num_classes)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        out = self.network(x)
        return out


def create_resnet18(num_classes=10, pretrained=True):
    """
    创建 ResNet-18 模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        
    Returns:
        model: ResNet-18 模型
    """
    # 加载预训练的 ResNet-18
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    
    # 修改最后一层全连接层以适配 CIFAR-10 的 10 分类
    # ResNet-18 的最后一个全连接层是 fc，输入特征数为 512
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, num_classes)
    )
    
    # 将模型移动到指定设备
    model = model.to(DEVICE)
    
    print(f"\n[OK] 创建模型：ResNet-18" + (" (预训练)" if pretrained else ""))
    
    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] 总参数量：{total_params:,}")
    print(f"[OK] 可训练参数量：{trainable_params:,}")
    
    return model


def create_vgg11(num_classes=10, pretrained=True):
    """
    创建 VGG-11 模型
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        
    Returns:
        model: VGG-11 模型
    """
    weights = models.VGG11_Weights.DEFAULT if pretrained else None
    model = models.vgg11(weights=weights)
    
    # 修改分类器部分
    # VGG-11 的 classifier[6] 是最后的线性层，输入为 4096
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    model = model.to(DEVICE)
    
    print(f"\n[OK] 创建模型：VGG-11" + (" (预训练)" if pretrained else ""))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] 总参数量：{total_params:,}")
    print(f"[OK] 可训练参数量：{trainable_params:,}")
    
    return model


def create_mobilenet_v2(num_classes=10, pretrained=True):
    """
    创建 MobileNetV2 模型（轻量级）
    
    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        
    Returns:
        model: MobileNetV2 模型
    """
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    
    # 修改最后的分类器
    # MobileNetV2 的 classifier[1] 是线性层，输入为 1280
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    model = model.to(DEVICE)
    
    print(f"\n[OK] 创建模型：MobileNetV2 (轻量级)" + (" (预训练)" if pretrained else ""))
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[OK] 总参数量：{total_params:,}")
    print(f"[OK] 可训练参数量：{trainable_params:,}")
    
    return model


def create_model(model_type='resnet18', pretrained=True):
    """
    创建分类模型
    
    Args:
        model_type: 模型类型 
                    - 'nn': 三层神经网络
                    - 'resnet18': ResNet-18 (推荐)
                    - 'vgg11': VGG-11
                    - 'mobilenet_v2': MobileNetV2 (轻量级)
        pretrained: 是否使用预训练权重（仅对迁移学习模型有效）
        
    Returns:
        model: PyTorch 模型
    """
    model_dict = {
        'nn': lambda: NeuralNetworkClassifier().to(DEVICE),
        'resnet18': lambda: create_resnet18(num_classes=10, pretrained=pretrained),
        'vgg11': lambda: create_vgg11(num_classes=10, pretrained=pretrained),
        'mobilenet_v2': lambda: create_mobilenet_v2(num_classes=10, pretrained=pretrained),
    }
    
    if model_type not in model_dict:
        raise ValueError(f"不支持的模型类型：{model_type}\n支持的类型：{list(model_dict.keys())}")
    
    model = model_dict[model_type]()
    
    # 显示模型名称
    model_names = {
        'nn': '三层神经网络分类器',
        'resnet18': 'ResNet-18 (残差网络)',
        'vgg11': 'VGG-11',
        'mobilenet_v2': 'MobileNetV2 (轻量级)'
    }
    print(f"[OK] 创建模型：{model_names.get(model_type, model_type)}")
    
    return model


# ==================== 训练函数 ====================

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """
    训练一个 epoch
    
    Args:
        model: 模型
        train_loader: 训练集数据加载器
        criterion: 损失函数
        optimizer: 优化器
        epoch: 当前 epoch 数
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    """
    训练模型
    
    Args:
        model: 要训练的模型
        train_loader: 训练集数据加载器
        test_loader: 测试集数据加载器
        epochs: 训练轮数
        lr: 学习率
        
    Returns:
        train_history: 训练历史记录
    """
    print("\n" + "="*70)
    print("开始训练模型")
    print("="*70)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 使用 Adam 优化器（比 SGD 更好）
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 使用 CosineAnnealing 学习率调度器（更平滑的衰减）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        print(f"\n[Epoch {epoch}/{epochs}]")
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch
        )
        
        # 更新学习率
        scheduler.step()
        
        # 测试
        test_acc = evaluate_model(model, test_loader)
        
        # 保存历史
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['test_acc'].append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint(model, optimizer, epoch, test_acc, is_best=True)
        
        print(f"\n训练损失：{train_loss:.4f} | 训练准确率：{train_acc:.2f}% | 测试准确率：{test_acc:.2f}%")
    
    print(f"\n[OK] 训练完成！最佳测试准确率：{best_acc:.2f}%")
    
    return train_history


# ==================== 测试验证函数 ====================

@torch.no_grad()
def evaluate_model(model, test_loader):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试集数据加载器
        
    Returns:
        accuracy: 测试准确率
    """
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def test_model(model, test_loader, class_names):
    """
    测试模型并显示详细结果
    
    Args:
        model: 模型
        test_loader: 测试集数据加载器
        class_names: 类别名称列表
        
    Returns:
        overall_accuracy: 总体准确率
        class_accuracies: 各类别准确率
    """
    print("\n" + "="*70)
    print("模型测试")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    
    # 用于统计每个类别的准确情况
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    all_preds = []
    all_labels = []
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 统计每个类别
        for i in range(targets.size(0)):
            label = targets[i].item()
            pred = predicted[i].item()
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
            
            all_preds.append(pred)
            all_labels.append(label)
    
    overall_accuracy = 100. * correct / total
    
    print(f"\n总体准确率：{overall_accuracy:.2f}% ({correct}/{total})")
    print("\n各类别准确率:")
    
    # 构建各类别准确率字典
    class_accuracies = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            class_accuracies[name] = acc
            print(f"  {name:10s}: {acc:6.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    return overall_accuracy, class_accuracies


# ==================== 工具函数 ====================

def save_checkpoint(model, optimizer, epoch, accuracy, filename='checkpoint.pth', is_best=False):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前 epoch
        accuracy: 当前准确率
        filename: 文件名
        is_best: 是否为最佳模型
    """
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
    }
    
    # 保存检查点
    checkpoint_path = os.path.join(OUTPUT_DIR, filename)
    torch.save(checkpoint, checkpoint_path)
    print(f"[OK] 保存检查点：{filename}")
    
    # 如果是最佳模型，额外保存一份
    if is_best:
        best_path = os.path.join(OUTPUT_DIR, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"[OK] 保存最佳模型：best_model.pth (准确率：{accuracy:.2f}%)")
    return checkpoint_path


def load_checkpoint(model, checkpoint_path):
    """
    加载模型检查点
    
    Args:
        model: 模型
        checkpoint_path: 检查点路径
        
    Returns:
        epoch: 加载的 epoch
        accuracy: 加载时的准确率
    """
    if not os.path.exists(checkpoint_path):
        print(f"警告：检查点文件不存在 - {checkpoint_path}")
        return 0, 0.0
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    accuracy = checkpoint.get('accuracy', 0.0)
    
    print(f"[OK] 加载检查点：{os.path.basename(checkpoint_path)} (Epoch {epoch}, 准确率：{accuracy:.2f}%)")
    
    return epoch, accuracy


def plot_training_history(history, save_path=None):
    """
    绘制训练历史曲线
    
    Args:
        history: 训练历史记录
        save_path: 保存路径
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 损失曲线
        ax1.plot(history['train_loss'], 'b-', label='训练损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('训练损失曲线')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_acc'], 'b-', label='训练准确率')
        ax2.plot(history['test_acc'], 'r-', label='测试准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('准确率曲线')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[OK] 保存训练历史图：{os.path.basename(save_path)}")
        
        plt.show()
        
    except ImportError:
        print("警告：matplotlib 未安装，无法绘制训练曲线")


# ==================== 主程序 ====================

def main():
    """
    主函数：执行完整的 CIFAR-10 分类流程
    """
    print("="*70)
    print("      CIFAR-10 图像分类实验 - 深度学习模型对比")
    print("="*70)
    
    # ==================== 配置参数 ====================
    DATA_ROOT = r'.\Data\inputs'
    BATCH_SIZE = 100
    EPOCHS = 15
    LEARNING_RATE = 0.001
    MODEL_TYPE = 'resnet18'  # 可选：'nn', 'resnet18'(推荐), 'vgg11', 'mobilenet_v2'
    USE_PRETRAINED = True  # 是否使用预训练权重（强烈推荐设为 True）
    
    # ==================== 步骤 1: 加载数据 ====================
    train_loader, test_loader, class_names = load_cifar10_data(
        data_root=DATA_ROOT,
        batch_size=BATCH_SIZE
    )
    
    # ==================== 步骤 2: 创建模型 ====================
    model = create_model(model_type=MODEL_TYPE, pretrained=USE_PRETRAINED)
    
    # ==================== 步骤 3: 训练模型 ====================
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )
    
    # ==================== 步骤 4: 测试模型 ====================
    overall_acc, class_accs = test_model(model, test_loader, class_names)
    
    # ==================== 步骤 5: 保存训练历史可视化 ====================
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plot_training_history(
        history,
        save_path=os.path.join(OUTPUT_DIR, 'training_history.png')
    )
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print(f"\n【输出目录】")
    print(f"[OK] 所有结果已保存到：{OUTPUT_DIR}")
    print(f"  - best_model.pth: 最佳模型权重")
    print(f"  - checkpoint.pth: 最新检查点")
    print(f"  - training_history.png: 训练曲线图")
    
    print(f"\n【最终结果】")
    print(f"[OK] 总体测试准确率：{overall_acc:.2f}%")
    print(f"[OK] 最佳测试准确率：{max(history['test_acc']):.2f}%")
    
    print(f"\n【模型信息】")
    print(f"[OK] 使用模型：{MODEL_TYPE}" + (" (预训练)" if USE_PRETRAINED else ""))
    print(f"[OK] 训练轮数：{EPOCHS}")
    print(f"[OK] Batch Size: {BATCH_SIZE}")


if __name__ == "__main__":
    main()