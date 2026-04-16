"""
使用 OpenCV 的 HOGDescriptor 和 SVM 进行图像分类
支持训练、保存/加载模型、预测
"""

import cv2
import numpy as np
import os
import pickle
from typing import Tuple
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearnex import patch_sklearn
patch_sklearn()   # 自动加速 LinearSVC（底层替换为多线程实现）


def imread_unicode(path: str) -> np.ndarray:
    """支持中文路径的 cv2.imread 替代函数"""
    try:
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
        np_array = np.asarray(bytes_data, dtype=np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class HOGSVMClassifier:
    def __init__(self, win_size=(64, 128), cell_size=(8, 8), block_size=(16, 16),
                 block_stride=(8, 8), nbins=9):
        """
        初始化 HOG 特征提取器 (OpenCV) 和 SVM 分类器
        """
        self.win_size = win_size
        self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride,
                                     cell_size, nbins)
        self.scaler = StandardScaler()
        self.svm = LinearSVC(C=1.0, max_iter=10000)
        self.class_names = []

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        提取图像的 HOG 特征 (OpenCV)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, self.win_size)
        features = self.hog.compute(resized)
        return features.flatten()

    def load_data_from_folders(self, root_dir: str, extensions=('.jpg', '.png', '.jpeg', '.pgm')):
        """
        从文件夹结构加载数据，每个子文件夹代表一个类别
        文件夹名即为类别名
        :return: X (特征列表), y (标签列表), class_names (类别名列表)
        """
        X = []
        y = []
        class_names = sorted([d for d in os.listdir(root_dir)
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_names = class_names

        for label, class_name in enumerate(class_names):
            class_path = os.path.join(root_dir, class_name)
            for file in os.listdir(class_path):
                if file.lower().endswith(extensions):
                    img_path = os.path.join(class_path, file)
                    img = imread_unicode(img_path)      # 已修改：支持中文路径
                    if img is not None:
                        feat = self.extract_features(img)
                        X.append(feat)
                        y.append(label)

        return np.array(X), np.array(y), class_names

    def train(self, X, y, test_size=0.2, random_state=42):
        """
        训练 SVM 分类器
        """
        if len(X) == 0:
            raise ValueError("No training data provided.")

        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self.svm.fit(X_train, y_train)

        y_pred = self.svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Training completed. Test Accuracy: {acc:.4f}")

        return acc

    def predict(self, image: np.ndarray) -> Tuple[int, str, float]:
        """
        对单张图像进行预测
        :return: (类别索引, 类别名, 置信度)
        """
        feat = self.extract_features(image)
        feat_scaled = self.scaler.transform([feat])
        decision = self.svm.decision_function(feat_scaled)

        if len(decision.shape) == 1:
            # 二分类
            if decision[0] > 0:
                pred_idx = 1
                confidence = decision[0]
            else:
                pred_idx = 0
                confidence = -decision[0]
        else:
            pred_idx = np.argmax(decision)
            confidence = decision[0, pred_idx]

        return pred_idx, self.class_names[pred_idx], confidence

    def save_model(self, filepath: str):
        """保存模型 (SVM + Scaler + 类别名)"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'scaler': self.scaler,
                'class_names': self.class_names,
                'win_size': self.win_size
            }, f)

    def load_model(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.svm = data['svm']
        self.scaler = data['scaler']
        self.class_names = data['class_names']
        self.win_size = data['win_size']
        self.hog = cv2.HOGDescriptor(self.win_size, (16,16), (8,8), (8,8), 9)