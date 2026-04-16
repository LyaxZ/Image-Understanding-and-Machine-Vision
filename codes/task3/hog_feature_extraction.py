"""
纯 Python 实现 HOG (Histogram of Oriented Gradients) 特征提取
完全手动实现梯度计算、cell 直方图、块归一化等步骤
参考 Dalal-Triggs 论文
"""

import numpy as np
import cv2
from typing import Tuple, List


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


class HOGFeatureExtractor:
    def __init__(self, cell_size: Tuple[int, int] = (8, 8),
                 block_size: Tuple[int, int] = (2, 2),
                 nbins: int = 9,
                 signed_gradient: bool = False):
        """
        :param cell_size: 细胞单元大小 (pixels)
        :param block_size: 块大小，以 cell 为单位
        :param nbins: 方向直方图的 bins 数量
        :param signed_gradient: 是否使用有符号梯度 (0-360度)，否则无符号 (0-180度)
        """
        self.cell_size = cell_size
        self.block_size = block_size
        self.nbins = nbins
        self.signed_gradient = signed_gradient

    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算图像的梯度幅值和方向
        :param image: 灰度图 (float64)
        :return: (magnitude, orientation) 方向角度范围 [0, 180) 或 [0, 360)
        """
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=1)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=1)

        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.rad2deg(np.arctan2(gy, gx))

        if not self.signed_gradient:
            orientation = np.abs(orientation)
        else:
            orientation[orientation < 0] += 360.0

        return magnitude, orientation

    def _cell_histogram(self, cell_mag: np.ndarray, cell_ori: np.ndarray) -> np.ndarray:
        """
        为单个 cell 计算方向直方图
        :param cell_mag: cell 内梯度幅值
        :param cell_ori: cell 内梯度方向
        :return: 长度为 nbins 的直方图
        """
        hist = np.zeros(self.nbins, dtype=np.float64)
        bin_width = 180.0 / self.nbins if not self.signed_gradient else 360.0 / self.nbins

        for i in range(cell_mag.shape[0]):
            for j in range(cell_mag.shape[1]):
                mag = cell_mag[i, j]
                ori = cell_ori[i, j]

                bin_idx = ori / bin_width
                bin_low = int(np.floor(bin_idx)) % self.nbins
                bin_high = (bin_low + 1) % self.nbins
                weight_high = bin_idx - bin_low
                weight_low = 1.0 - weight_high

                hist[bin_low] += mag * weight_low
                hist[bin_high] += mag * weight_high

        return hist

    def compute_hog(self, image: np.ndarray, visualize: bool = False) -> np.ndarray:
        """
        计算整幅图像的 HOG 特征向量
        :param image: 输入图像 (BGR 或灰度)，内部会转为灰度图
        :param visualize: 是否返回可视化图像 (暂不支持，仅返回特征)
        :return: 归一化后的 HOG 特征向量 (1D)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = gray.astype(np.float64) / 255.0

        h, w = gray.shape
        cell_h, cell_w = self.cell_size
        block_h, block_w = self.block_size

        mag, ori = self.compute_gradients(gray)

        n_cells_x = w // cell_w
        n_cells_y = h // cell_h

        cell_histograms = np.zeros((n_cells_y, n_cells_x, self.nbins), dtype=np.float64)

        for i in range(n_cells_y):
            for j in range(n_cells_x):
                y0, y1 = i * cell_h, (i + 1) * cell_h
                x0, x1 = j * cell_w, (j + 1) * cell_w
                cell_mag = mag[y0:y1, x0:x1]
                cell_ori = ori[y0:y1, x0:x1]
                cell_histograms[i, j, :] = self._cell_histogram(cell_mag, cell_ori)

        n_blocks_y = n_cells_y - block_h + 1
        n_blocks_x = n_cells_x - block_w + 1
        block_features = []

        eps = 1e-6
        for i in range(n_blocks_y):
            for j in range(n_blocks_x):
                block = cell_histograms[i:i+block_h, j:j+block_w, :].flatten()
                norm = np.sqrt(np.sum(block**2) + eps**2)
                block_norm = block / norm
                block_features.append(block_norm)

        if len(block_features) == 0:
            return np.array([])

        hog_features = np.concatenate(block_features)
        return hog_features


def render_hog(image: np.ndarray, hog_extractor: HOGFeatureExtractor) -> np.ndarray:
    """
    绘制 HOG 特征可视化图（简化版，直接返回原图）
    """
    # 可扩展，此处返回原图示意
    return image