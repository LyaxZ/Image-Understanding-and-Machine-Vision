"""
基于颜色直方图或HOG特征的图像检索模块
支持颜色直方图、LBP纹理、HOG特征（可选）
包含直方图可视化功能
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from skimage.feature import local_binary_pattern
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


class ColorHistogramRetriever:
    def __init__(self, bins=(8, 12, 3), feature_type='auto',
                 use_face_detection=False, distance_metric='chisqr',
                 hog_win_size=(64, 128), hog_cell_size=(8, 8),
                 hog_block_size=(16, 16), hog_block_stride=(8, 8), hog_nbins=9):
        """
        :param bins: HSV 直方图的 bins 数量
        :param feature_type: 特征类型
            - 'auto': 自动选择（灰度图用LBP，彩色图用HSV）
            - 'lbp': LBP纹理直方图
            - 'gray': 灰度直方图
            - 'hsv': HSV彩色直方图
            - 'hog': HOG特征（梯度直方图）
        :param use_face_detection: 是否启用人脸检测裁剪
        :param distance_metric: 距离度量方式 'correl'(相关性), 'chisqr'(卡方), 'bhatta'(巴氏), 'euclidean'(欧氏)
        :param hog_win_size, ... : 当 feature_type='hog' 时的 HOG 参数
        """
        self.bins = bins
        self.feature_type = feature_type
        self.use_face_detection = use_face_detection
        self.distance_metric = distance_metric
        self.hog_win_size = hog_win_size
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size
        self.hog_block_stride = hog_block_stride
        self.hog_nbins = hog_nbins

        self.hog = None
        if feature_type == 'hog':
            self.hog = cv2.HOGDescriptor(
                hog_win_size, hog_block_size, hog_block_stride,
                hog_cell_size, hog_nbins
            )

        self.database_features = {}
        self.database_paths = []

    def _preprocess_face(self, image: np.ndarray) -> np.ndarray:
        if not self.use_face_detection:
            return image
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            return image[y:y+h, x:x+w]
        return image

    def _extract_lbp_feature(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float32)
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def _extract_hog_feature(self, image: np.ndarray) -> np.ndarray:
        if self.hog is None:
            self.hog = cv2.HOGDescriptor(
                self.hog_win_size, self.hog_block_size, self.hog_block_stride,
                self.hog_cell_size, self.hog_nbins
            )
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        resized = cv2.resize(gray, self.hog_win_size)
        features = self.hog.compute(resized).flatten()
        # L2 归一化
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features

    def extract_feature(self, image: np.ndarray) -> np.ndarray:
        image = self._preprocess_face(image)

        if self.feature_type == 'auto':
            if len(image.shape) == 2 or image.shape[2] == 1:
                return self._extract_lbp_feature(image)
            else:
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins,
                                    [0, 180, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                return hist
        elif self.feature_type == 'lbp':
            return self._extract_lbp_feature(image)
        elif self.feature_type == 'gray':
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        elif self.feature_type == 'hsv':
            if len(image.shape) == 2 or image.shape[2] == 1:
                raise ValueError("HSV feature requires color image.")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, self.bins,
                                [0, 180, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        elif self.feature_type == 'hog':
            return self._extract_hog_feature(image)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")

    def build_database(self, folder_path: str):
        self.database_features.clear()
        self.database_paths.clear()
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.pgm')):
                    path = os.path.join(root, file)
                    img = imread_unicode(path)
                    if img is not None:
                        feat = self.extract_feature(img)
                        self.database_features[path] = feat
                        self.database_paths.append(path)
        print(f"Database built: {len(self.database_paths)} images indexed.")

    def retrieve(self, query_img_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        if not self.database_features:
            raise ValueError("Database is empty. Please build database first.")

        query_img = imread_unicode(query_img_path)
        if query_img is None:
            raise FileNotFoundError(f"Cannot read query image: {query_img_path}")

        query_feat = self.extract_feature(query_img)

        similarities = []
        if self.distance_metric == 'correl':
            cmp_method = cv2.HISTCMP_CORREL
            for path, feat in self.database_features.items():
                sim = cv2.compareHist(query_feat, feat, cmp_method)
                similarities.append((path, sim))
        elif self.distance_metric == 'chisqr':
            for path, feat in self.database_features.items():
                dist = cv2.compareHist(query_feat, feat, cv2.HISTCMP_CHISQR)
                sim = 1.0 / (1.0 + dist)
                similarities.append((path, sim))
        elif self.distance_metric == 'bhatta':
            for path, feat in self.database_features.items():
                dist = cv2.compareHist(query_feat, feat, cv2.HISTCMP_BHATTACHARYYA)
                sim = 1.0 / (1.0 + dist)
                similarities.append((path, sim))
        elif self.distance_metric == 'euclidean':
            for path, feat in self.database_features.items():
                dist = np.linalg.norm(query_feat - feat)
                sim = 1.0 / (1.0 + dist)
                similarities.append((path, sim))
        else:
            raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_feature_visualization(self, image: np.ndarray) -> np.ndarray:
        """
        根据当前特征类型，返回特征的可视化图像
        """
        image = self._preprocess_face(image)
        h, w = 400, 500  # 输出画布尺寸

        # ---- 根据特征类型绘制 ----
        if self.feature_type == 'gray' or (self.feature_type == 'auto' and (len(image.shape) == 2 or image.shape[2] == 1)):
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            return self._draw_single_histogram(hist, title='Gray Histogram', color=(0,0,0), width=w, height=h)

        elif self.feature_type == 'hsv' or (self.feature_type == 'auto' and len(image.shape) == 3 and image.shape[2] == 3):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            return self._draw_multi_histogram([h_hist, s_hist, v_hist],
                                              titles=['Hue', 'Saturation', 'Value'],
                                              colors=[(255,0,0), (0,255,0), (0,0,255)],
                                              width=w, height=h)

        elif self.feature_type == 'lbp':
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            n_bins = n_points + 2
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype(np.float32)
            return self._draw_single_histogram(hist, title='LBP Histogram', color=(128,0,128), width=w, height=h)

        elif self.feature_type == 'hog':
            feat = self._extract_hog_feature(image)
            return self._draw_single_histogram(feat, title='HOG Feature Distribution', color=(0,128,128), width=w, height=h)

        else:
            blank = np.ones((h, w, 3), dtype=np.uint8) * 255
            cv2.putText(blank, "No visualization", (30, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            return blank

    def _draw_single_histogram(self, hist, title='Histogram', color=(0,0,0), width=500, height=400):
        """绘制单一直方图"""
        hist = hist.flatten()
        bin_count = len(hist)
        bin_w = max(1, int(width / bin_count))
        img = np.ones((height, width, 3), dtype=np.uint8) * 255

        if np.max(hist) > 0:
            hist_norm = hist / np.max(hist) * (height - 50)
        else:
            hist_norm = hist

        for i in range(bin_count):
            x = i * bin_w
            h_val = int(hist_norm[i])
            cv2.rectangle(img, (x, height - h_val), (x + bin_w - 1, height), color, -1)

        cv2.putText(img, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        return img

    def _draw_multi_histogram(self, hists, titles, colors, width=500, height=400):
        """绘制多个直方图叠加"""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        for idx, (hist, color, title) in enumerate(zip(hists, colors, titles)):
            hist = hist.flatten()
            bin_count = len(hist)
            bin_w = max(1, int(width / bin_count))
            if np.max(hist) > 0:
                hist_norm = hist / np.max(hist) * (height - 50)
            else:
                hist_norm = hist

            for i in range(bin_count):
                x = i * bin_w
                h_val = int(hist_norm[i])
                cv2.rectangle(img, (x, height - h_val), (x + bin_w - 1, height), color, -1)
            cv2.putText(img, title, (10, 20 + idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        return img