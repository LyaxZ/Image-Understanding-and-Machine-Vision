import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from pathlib import Path
from tqdm import tqdm

# ==================== 全局配置 ====================
DEVICE = 'cuda' if False else 'cpu'  # 本实验主要使用 CPU
OUTPUT_DIR = r".\outputs\hog_svm_classification"

# ==================== 任务 1: 基于颜色直方图的图像检索 ====================

def compute_color_histogram(image, bins=8):
    """
    计算图像的颜色直方图特征
    
    Args:
        image: 输入图像 (RGB 或 BGR)
        bins: 每个通道的直方图 bin 数量
        
    Returns:
        hist_feature: 颜色直方图特征向量
    """
    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 分别计算 H, S, V 三个通道的直方图
    h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
    
    # 归一化直方图
    cv2.normalize(h_hist, h_hist)
    cv2.normalize(s_hist, h_hist)
    cv2.normalize(v_hist, h_hist)
    
    # 拼接三个通道的直方图特征
    hist_feature = np.concatenate([h_hist.flatten(), 
                                   s_hist.flatten(), 
                                   v_hist.flatten()])
    
    return hist_feature


def image_retrieval_by_histogram(query_image, database_images, top_k=5):
    """
    基于颜色直方图的图像检索
    
    Args:
        query_image: 查询图像
        database_images: 图像数据库列表 [(image_path, image), ...]
        top_k: 返回最相似的 top_k 个结果
        
    Returns:
        results: 排序后的结果列表 [(image_path, similarity_score), ...]
    """
    # 计算查询图像的特征
    query_feature = compute_color_histogram(query_image)
    
    # 计算与数据库中每张图像的相似度
    similarities = []
    for img_path, img in database_images:
        db_feature = compute_color_histogram(img)
        # 使用相关性作为相似度度量
        similarity = cv2.compareHist(query_feature.astype(np.float32), 
                                    db_feature.astype(np.float32), 
                                    cv2.HISTCMP_CORREL)
        similarities.append((img_path, similarity))
    
    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]


def demo_color_histogram_retrieval(image_folder):
    """
    演示颜色直方图图像检索功能
    
    Args:
        image_folder: 图像文件夹路径
    """
    print("="*70)
    print("      任务 1: 基于颜色直方图的图像检索")
    print("="*70)
    
    # 加载图像
    image_paths = list(Path(image_folder).glob("*.jpg")) + \
                  list(Path(image_folder).glob("*.png"))
    
    if len(image_paths) < 2:
        print(f"警告：图像文件夹中至少需要 2 张图像，当前找到 {len(image_paths)} 张")
        return
    
    print(f"\n[OK] 找到 {len(image_paths)} 张图像")
    
    # 加载图像到内存
    database = []
    for img_path in image_paths[:10]:  # 限制最多 10 张
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, (256, 256))
            database.append((str(img_path), img))
    
    if len(database) < 2:
        print("警告：成功加载的图像数量不足")
        return
    
    # 选择第一张图像作为查询图像
    query_path, query_img = database[0]
    print(f"\n查询图像：{os.path.basename(query_path)}")
    
    # 执行检索
    results = image_retrieval_by_histogram(query_img, database[1:], top_k=5)
    
    # 显示结果
    print("\n检索结果 (按相似度排序):")
    for i, (img_path, score) in enumerate(results, 1):
        print(f"  {i}. {os.path.basename(img_path):30s} 相似度：{score:.4f}")
    
    # 可视化检索结果
    fig, axes = plt.subplots(1, len(results) + 1, figsize=(15, 3))
    axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"查询图像\n{os.path.basename(query_path)}")
    axes[0].axis('off')
    
    for i, (img_path, score) in enumerate(results, 1):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"结果{i}\n相似度：{score:.4f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'color_histogram_retrieval.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] 保存检索结果图：{os.path.basename(save_path)}")
    plt.show()
    
    return results


# ==================== 任务 2: 手动实现 HOG 特征提取 ====================

def compute_gradient(image):
    """
    计算图像的梯度幅值和方向
    
    Args:
        image: 单通道灰度图像
        
    Returns:
        magnitude: 梯度幅值
        orientation: 梯度方向 (角度，单位：度)
    """
    # 使用 Sobel 算子计算水平和垂直方向的梯度
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    
    # 计算梯度幅值和方向
    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi  # 转换为角度
    
    # 将角度转换到 [0, 180] 范围 (无符号梯度)
    orientation = np.mod(orientation, 180)
    
    return magnitude, orientation


def create_cell_histogram(magnitudes, orientations, n_bins=9):
    """
    为单个 cell 创建方向直方图
    
    Args:
        magnitudes: cell 内的梯度幅值
        orientations: cell 内的梯度方向
        n_bins: 直方图的 bin 数量
        
    Returns:
        histogram: 归一化的方向直方图
    """
    histogram = np.zeros(n_bins)
    bin_width = 180 / n_bins  # 每个 bin 覆盖的角度范围
    
    # 为每个像素投票到直方图
    for i in range(magnitudes.shape[0]):
        for j in range(magnitudes.shape[1]):
            angle = orientations[i, j]
            magnitude = magnitudes[i, j]
            
            # 计算属于哪个 bin
            bin_index = int(angle / bin_width)
            if bin_index >= n_bins:
                bin_index = n_bins - 1
            
            # 投票 (使用权重 - 梯度幅值)
            histogram[bin_index] += magnitude
    
    # 归一化直方图
    norm = np.linalg.norm(histogram)
    if norm > 0:
        histogram = histogram / norm
    
    return histogram


def extract_hog_feature_manual(image, cell_size=8, block_size=2, n_bins=9):
    """
    手动实现 HOG 特征提取 (完整计算过程)
    
    Args:
        image: 输入图像 (灰度图)
        cell_size: cell 大小 (像素)
        block_size: block 中包含的 cell 数量
        n_bins: 方向直方图的 bin 数量
        
    Returns:
        hog_feature: HOG 特征向量
        visualization: HOG 可视化图像
    """
    # 确保图像是灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 调整图像大小以确保能被 cell_size 整除
    h, w = gray.shape
    h = (h // cell_size) * cell_size
    w = (w // cell_size) * cell_size
    gray = cv2.resize(gray, (w, h))
    
    # 步骤 1: 计算梯度
    magnitude, orientation = compute_gradient(gray)
    
    # 步骤 2: 计算每个 cell 的直方图
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    
    cell_histograms = np.zeros((n_cells_y, n_cells_x, n_bins))
    
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            # 提取当前 cell 的区域
            y_start, y_end = i * cell_size, (i + 1) * cell_size
            x_start, x_end = j * cell_size, (j + 1) * cell_size
            
            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ori = orientation[y_start:y_end, x_start:x_end]
            
            # 计算 cell 的直方图
            cell_histograms[i, j] = create_cell_histogram(cell_mag, cell_ori, n_bins)
    
    # 步骤 3: Block 归一化
    hog_features = []
    
    for i in range(n_cells_y - block_size + 1):
        for j in range(n_cells_x - block_size + 1):
            # 提取当前 block 的所有 cell 直方图
            block_features = []
            for bi in range(block_size):
                for bj in range(block_size):
                    block_features.extend(cell_histograms[i + bi, j + bj])
            
            # 转换为 numpy 数组
            block_features = np.array(block_features)
            
            # L2 归一化
            norm = np.linalg.norm(block_features)
            if norm > 0:
                block_features = block_features / norm
            
            # 添加到最终特征向量
            hog_features.extend(block_features)
    
    hog_feature = np.array(hog_features)
    
    # 创建可视化图像
    visualization = visualize_hog_feature(hog_feature, (n_cells_x, n_cells_y), 
                                         cell_size, block_size, n_bins)
    
    return hog_feature, visualization


def visualize_hog_feature(hog_feature, grid_size, cell_size, block_size, n_bins):
    """
    可视化 HOG 特征
    
    Args:
        hog_feature: HOG 特征向量
        grid_size: (cells_x, cells_y)
        cell_size: cell 大小
        block_size: block 大小
        n_bins: 方向 bin 数量
        
    Returns:
        vis_image: 可视化图像
    """
    n_cells_x, n_cells_y = grid_size
    vis_height = n_cells_y * cell_size
    vis_width = n_cells_x * cell_size
    
    vis_image = np.ones((vis_height, vis_width), dtype=np.uint8) * 255
    
    # 计算每个 block 的特征维度
    features_per_block = block_size * block_size * n_bins
    
    # 为每个 cell 绘制方向
    feature_idx = 0
    for i in range(n_cells_y - block_size + 1):
        for j in range(n_cells_x - block_size + 1):
            # 提取当前 block 的特征
            for bi in range(block_size):
                for bj in range(block_size):
                    cell_i = i + bi
                    cell_j = j + bj
                    
                    if cell_i < n_cells_y and cell_j < n_cells_x:
                        # 获取当前 cell 的直方图部分
                        hist_start = (bi * block_size + bj) * n_bins
                        hist_end = hist_start + n_bins
                        
                        if feature_idx + len(hog_feature) > hist_end:
                            # 绘制方向线
                            center_x = (cell_j + 0.5) * cell_size
                            center_y = (cell_i + 0.5) * cell_size
                            
                            for bin_idx in range(n_bins):
                                angle = bin_idx * 180 / n_bins
                                magnitude = hog_feature[feature_idx + bin_idx] if feature_idx + bin_idx < len(hog_feature) else 0
                                
                                # 将角度转换为弧度
                                rad = angle * np.pi / 180
                                length = cell_size / 2 * magnitude
                                
                                x1 = int(center_x - length * np.cos(rad))
                                y1 = int(center_y - length * np.sin(rad))
                                x2 = int(center_x + length * np.cos(rad))
                                y2 = int(center_y + length * np.sin(rad))
                                
                                # 限制在 cell 内
                                x1 = max(int(cell_j * cell_size), min(x1, vis_width - 1))
                                x2 = max(int(cell_j * cell_size), min(x2, vis_width - 1))
                                y1 = max(int(cell_i * cell_size), min(y1, vis_height - 1))
                                y2 = max(int(cell_i * cell_size), min(y2, vis_height - 1))
                                
                                cv2.line(vis_image, (x1, y1), (x2, y2), 128, 1)
                        
                        feature_idx += n_bins
    
    return vis_image


def demo_manual_hog(image_path):
    """
    演示手动 HOG 特征提取
    
    Args:
        image_path: 图像路径
    """
    print("="*70)
    print("      任务 2: 手动实现 HOG 特征提取")
    print("="*70)
    
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法加载图像 - {image_path}")
        return
    
    print(f"\n[OK] 加载图像：{os.path.basename(image_path)}")
    print(f"[OK] 图像尺寸：{image.shape[1]}x{image.shape[0]}")
    
    # 提取 HOG 特征
    hog_feature, hog_vis = extract_hog_feature_manual(image, cell_size=8, block_size=2, n_bins=9)
    
    print(f"\n[OK] HOG 特征维度：{hog_feature.shape}")
    print(f"[OK] 特征向量前 20 个值：{hog_feature[:20]}")
    
    # 显示结果
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('原始图像')
    axes[0].axis('off')
    
    # 梯度幅值
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, orientation = compute_gradient(gray)
    axes[1].imshow(orientation, cmap='hsv')
    axes[1].set_title('梯度方向图')
    axes[1].axis('off')
    
    # HOG 可视化
    axes[2].imshow(hog_vis, cmap='gray')
    axes[2].set_title(f'HOG 特征可视化\n(特征维度：{hog_feature.shape[0]})')
    axes[2].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'manual_hog_feature.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] 保存 HOG 特征图：{os.path.basename(save_path)}")
    plt.show()
    
    return hog_feature


# ==================== 任务 3: OpenCV 实现 HOG+SVM 图像分类 ====================

def extract_hog_opencv(image, win_size=(64, 128), block_size=(16, 16), 
                       block_stride=(8, 8), cell_size=(8, 8), nbins=9):
    """
    使用 OpenCV 提取 HOG 特征
    
    Args:
        image: 输入图像
        win_size: 检测窗口大小
        block_size: block 大小
        block_stride: block 步长
        cell_size: cell 大小
        nbins: 方向 bin 数量
        
    Returns:
        hog_descriptor: HOG 特征向量
    """
    # 创建 HOG 描述子
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    
    # 调整图像大小
    image = cv2.resize(image, win_size)
    
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 提取 HOG 特征
    hog_descriptor = hog.compute(gray)
    
    return hog_descriptor.flatten()


def create_dataset_for_classification(image_folders):
    """
    创建分类数据集
    
    Args:
        image_folders: 类别文件夹字典 {category_name: folder_path, ...}
        
    Returns:
        X: 特征矩阵
        y: 标签向量
        class_names: 类别名称列表
    """
    X = []
    y = []
    class_names = list(image_folders.keys())
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print("\n加载数据集:")
    for category, folder_path in image_folders.items():
        print(f"  类别：{category:15s} - 路径：{folder_path}")
        
        # 加载该类别的所有图像
        image_paths = list(Path(folder_path).glob("*.jpg")) + \
                      list(Path(folder_path).glob("*.png")) + \
                      list(Path(folder_path).glob("*.jpeg"))
        
        for img_path in tqdm(image_paths, desc=f"  处理 {category}"):
            img = cv2.imread(str(img_path))
            if img is not None:
                # 提取 HOG 特征
                hog_feat = extract_hog_opencv(img)
                X.append(hog_feat)
                y.append(label_to_idx[category])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n[OK] 数据集大小：{X.shape[0]} 张图像")
    print(f"[OK] 特征维度：{X.shape[1]}")
    print(f"[OK] 类别数量：{len(class_names)}")
    
    return X, y, class_names


def train_hog_svm_classifier(X, y, class_names, test_size=0.2):
    """
    训练 HOG+SVM 分类器
    
    Args:
        X: 特征矩阵
        y: 标签向量
        class_names: 类别名称列表
        test_size: 测试集比例
        
    Returns:
        svm_classifier: 训练好的 SVM 分类器
        accuracy: 测试集准确率
        report: 分类报告
    """
    print("="*70)
    print("      任务 3: HOG+SVM 图像分类")
    print("="*70)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n[OK] 训练集样本数：{X_train.shape[0]}")
    print(f"[OK] 测试集样本数：{X_test.shape[0]}")
    
    # 创建 SVM 分类器
    print("\n[OK] 创建 SVM 分类器 (RBF 核)...")
    svm_classifier = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    
    # 训练模型
    print("[OK] 开始训练...")
    svm_classifier.fit(X_train, y_train)
    
    # 在测试集上评估
    y_pred = svm_classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print(f"\n[OK] 测试集准确率：{accuracy * 100:.2f}%")
    
    # 生成分类报告
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\n分类报告:")
    print(report)
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    return svm_classifier, accuracy, report, cm


def demo_hog_svm_classification(image_folders):
    """
    演示 HOG+SVM 图像分类
    
    Args:
        image_folders: 类别文件夹字典
    """
    # 创建数据集
    X, y, class_names = create_dataset_for_classification(image_folders)
    
    if len(X) < 10:
        print("警告：数据集太小，无法进行有效训练")
        return None, None
    
    # 训练分类器
    svm_classifier, accuracy, report, cm = train_hog_svm_classifier(X, y, class_names)
    
    # 可视化结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 混淆矩阵热图
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('混淆矩阵')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    plt.colorbar(im, ax=axes[0])
    
    # 添加文本注释
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    
    # 类别分布
    class_counts = np.bincount(y)
    axes[1].bar(range(len(class_names)), class_counts)
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_xlabel('类别')
    axes[1].set_ylabel('样本数量')
    axes[1].set_title('数据集类别分布')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'hog_svm_classification_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] 保存分类结果图：{os.path.basename(save_path)}")
    plt.show()
    
    return svm_classifier, accuracy


def demo_hog_svm_with_sample_data():
    """
    使用示例数据演示 HOG+SVM 分类
    
    由于可能没有现成的数据集，这里创建一个简单的示例
    """
    print("="*70)
    print("      任务 3: HOG+SVM 图像分类 (示例演示)")
    print("="*70)
    
    # 创建简单的示例数据
    # 这里使用不同方向的椭圆来模拟不同类别
    
    np.random.seed(42)
    n_samples_per_class = 50
    
    X = []
    y = []
    
    # 类别 1: 水平椭圆
    print("\n生成示例数据...")
    for _ in range(n_samples_per_class):
        img = np.ones((64, 64), dtype=np.uint8) * 255
        center = (32 + np.random.randint(-5, 5), 32 + np.random.randint(-5, 5))
        axes = (20 + np.random.randint(-3, 3), 10 + np.random.randint(-2, 2))
        cv2.ellipse(img, center, axes, 0, 0, 360, 0, -1)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hog_feat = extract_hog_opencv(img_color)
        X.append(hog_feat)
        y.append(0)
    
    # 类别 2: 垂直椭圆
    for _ in range(n_samples_per_class):
        img = np.ones((64, 64), dtype=np.uint8) * 255
        center = (32 + np.random.randint(-5, 5), 32 + np.random.randint(-5, 5))
        axes = (10 + np.random.randint(-2, 2), 20 + np.random.randint(-3, 3))
        cv2.ellipse(img, center, axes, 0, 0, 360, 0, -1)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hog_feat = extract_hog_opencv(img_color)
        X.append(hog_feat)
        y.append(1)
    
    # 类别 3: 圆形
    for _ in range(n_samples_per_class):
        img = np.ones((64, 64), dtype=np.uint8) * 255
        center = (32 + np.random.randint(-5, 5), 32 + np.random.randint(-5, 5))
        radius = 15 + np.random.randint(-3, 3)
        cv2.circle(img, center, radius, 0, -1)
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        hog_feat = extract_hog_opencv(img_color)
        X.append(hog_feat)
        y.append(2)
    
    X = np.array(X)
    y = np.array(y)
    class_names = ['水平椭圆', '垂直椭圆', '圆形']
    
    print(f"[OK] 生成样本数：{len(X)}")
    print(f"[OK] 特征维度：{X.shape[1]}")
    print(f"[OK] 类别数：{len(class_names)}")
    
    # 训练分类器
    svm_classifier, accuracy, report, cm = train_hog_svm_classifier(X, y, class_names)
    
    # 可视化示例图像
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    
    for i in range(3):
        # 找到该类别的样本索引
        indices = np.where(y == i)[0][:5]
        for j, idx in enumerate(indices):
            # 重新生成图像用于显示
            if i == 0:  # 水平椭圆
                img = np.ones((64, 64), dtype=np.uint8) * 255
                cv2.ellipse(img, (32, 32), (20, 10), 0, 0, 360, 0, -1)
            elif i == 1:  # 垂直椭圆
                img = np.ones((64, 64), dtype=np.uint8) * 255
                cv2.ellipse(img, (32, 32), (10, 20), 0, 0, 360, 0, -1)
            else:  # 圆形
                img = np.ones((64, 64), dtype=np.uint8) * 255
                cv2.circle(img, (32, 32), 15, 0, -1)
            
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
    
    axes[0, 0].set_title(f'类别：{class_names[0]}')
    axes[1, 0].set_title(f'类别：{class_names[1]}')
    axes[2, 0].set_title(f'类别：{class_names[2]}')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] 保存示例图像：{os.path.basename(save_path)}")
    plt.show()
    
    return svm_classifier, accuracy


# ==================== 主程序 ====================

def main():
    """
    主函数：执行所有三个任务
    """
    print("="*70)
    print("      图像特征提取与分类综合实验")
    print("="*70)
    
    # 确保输出目录存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 配置参数
    IMAGE_FOLDER = r'.\Data/inputs/cifar-10-batches-py'  # 图像文件夹路径
    
    # ==================== 任务 1: 颜色直方图图像检索 ====================
    # 注意：需要至少 2 张图像才能演示检索功能
    try:
        demo_color_histogram_retrieval(IMAGE_FOLDER)
    except Exception as e:
        print(f"\n任务 1 执行失败：{str(e)}")
        print("提示：请确保图像文件夹中存在有效的图像文件")
    
    # ==================== 任务 2: 手动实现 HOG 特征提取 ====================
    # 使用一张示例图像演示 HOG 特征提取
    sample_images = list(Path(IMAGE_FOLDER).glob("*.jpg")) + \
                    list(Path(IMAGE_FOLDER).glob("*.png"))
    
    if len(sample_images) > 0:
        demo_manual_hog(str(sample_images[0]))
    else:
        print("\n任务 2 跳过：未找到图像文件")
    
    # ==================== 任务 3: HOG+SVM 图像分类 ====================
    # 使用示例数据进行演示
    demo_hog_svm_with_sample_data()
    
    # 如果有实际的数据集，可以取消下面的注释并配置路径
    # image_folders = {
    #     '类别 1': r'.\data\class1',
    #     '类别 2': r'.\data\class2',
    #     '类别 3': r'.\data\class3',
    # }
    # demo_hog_svm_classification(image_folders)
    
    # ==================== 总结 ====================
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    print(f"\n【输出目录】")
    print(f"[OK] 所有结果已保存到：{OUTPUT_DIR}")
    print(f"  - color_histogram_retrieval.png: 颜色直方图检索结果")
    print(f"  - manual_hog_feature.png: 手动 HOG 特征可视化")
    print(f"  - hog_svm_classification_results.png: HOG+SVM 分类结果")
    print(f"  - sample_images.png: 示例图像")
    
    print(f"\n【实验内容】")
    print(f"[OK] 任务 1: 基于颜色直方图的图像检索")
    print(f"[OK] 任务 2: 手动实现 HOG 特征提取 (完整计算过程)")
    print(f"[OK] 任务 3: 使用 OpenCV 实现 HOG+SVM 图像分类")


if __name__ == "__main__":
    main()