import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QPushButton, QFileDialog, QGroupBox,
    QScrollArea, QSplitter, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont


# ============================================================
#  兼容中文路径
# ============================================================
def cv2_imread(path, flags=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)


def cv2_imwrite(path, img, params=None):
    ext = path.rsplit('.', 1)[-1].lower()
    ok, enc = cv2.imencode('.' + ext, img, params)
    if ok:
        enc.tofile(path)
        return True
    return False


# ============================================================
#  图像处理核心函数
# ============================================================
def adjust_brightness(img, value):
    if value == 0:
        return img
    return np.clip(img.astype(np.int16) + value, 0, 255).astype(np.uint8)


def adjust_contrast(img, value):
    if value == 0:
        return img
    factor = 1.0 + value / 50.0
    if factor < 0.1:
        factor = 0.1
    if factor > 1.0:
        factor = 1.0 + (factor - 1.0) * 2.5
    out = (img.astype(np.float32) - 128.0) * factor + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(img, value):
    if value == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + value / 100.0), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_gamma(img, g):
    if g == 1.0:
        return img
    table = np.array([((i / 255.0) ** (1.0 / g)) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def sharpen_image(img, amount):
    if amount == 0:
        return img
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    alpha = amount / 100.0
    return cv2.addWeighted(img, 1 - alpha,
                           cv2.filter2D(img, -1, kernel), alpha, 0)


# ==================== 新增均值滤波 ====================
def mean_filter(img, ksize):
    """均值滤波 - 计算邻域内像素的平均值"""
    if ksize <= 1:
        return img
    return cv2.blur(img, (ksize, ksize))


def gaussian_blur(img, ksize):
    if ksize <= 1:
        return img
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def median_blur(img, ksize):
    if ksize <= 1:
        return img
    return cv2.medianBlur(img, ksize)


def bilateral_filter(img, d):
    if d <= 1:
        return img
    return cv2.bilateralFilter(img, d, 75, 75)


def adjust_hue(img, value):
    if value == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + value) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def adjust_shadows(img, value):
    if value == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = L.astype(np.float32)
    mask = L < 128
    L[mask] = L[mask] * (1 + value / 200.0)
    L = np.clip(L, 0, 255)
    lab = cv2.merge([L.astype(np.uint8), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def adjust_highlights(img, value):
    if value == 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = L.astype(np.float32)
    mask = L > 128
    L[mask] = L[mask] * (1 + value / 200.0)
    L = np.clip(L, 0, 255)
    lab = cv2.merge([L.astype(np.uint8), a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ============================================================
#  绘制直方图
# ============================================================
def draw_rgb_histogram(img):
    hist_h, hist_w = 200, 512
    hist_image = np.full((hist_h, hist_w, 3), 50, dtype=np.uint8)
    
    bins = np.arange(256).reshape(256, 1)
    scaled_bins = (bins * hist_w / 256).astype(np.int32)
    
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    
    for ch, col in enumerate(colors):
        hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_h - 10, cv2.NORM_MINMAX)
        pts = np.column_stack((scaled_bins, hist_h - hist)).astype(np.int32)
        cv2.polylines(hist_image, [pts], False, col, 2, cv2.LINE_AA)
        
    return hist_image


# ============================================================
#  滑块组件
# ============================================================
class SliderControl(QWidget):
    def __init__(self, name, lo, hi, default, suffix="", parent=None):
        super().__init__(parent)
        self.default_val = default
        self.suffix = suffix
        self.value_changed_callback = None

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(10)

        self.label = QLabel(name)
        self.label.setFixedWidth(90)
        self.label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(default)
        self.slider.setFixedHeight(26)
        self.slider.valueChanged.connect(self._changed)

        self.val_lbl = QLabel(str(default) + suffix)
        self.val_lbl.setFixedWidth(60)
        self.val_lbl.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.rst = QPushButton("↺")
        self.rst.setFixedSize(32, 32)
        self.rst.setCursor(Qt.PointingHandCursor)
        self.rst.clicked.connect(lambda: self.slider.setValue(self.default_val))

        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val_lbl)
        lay.addWidget(self.rst)

        self.label.setStyleSheet(
            "color:#d0d0d0; font-size:15px; font-weight:bold;")
        self.val_lbl.setStyleSheet(
            "color:#00d4aa; font-size:14px; font-weight:bold;"
            "background:#1e1e2e; border-radius:6px; padding:3px 6px;")
        self.rst.setStyleSheet(
            "QPushButton{background:#2a2a3e;color:#ff6b6b;"
            "border:1px solid #3a3a4e;border-radius:8px;"
            "font-size:20px;font-weight:bold}"
            "QPushButton:hover{background:#3a3a5e;border:1px solid #ff6b6b}"
            "QPushButton:pressed{background:#ff6b6b;color:#fff}")
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{border:1px solid #3a3a4e;"
            "height:8px;background:#1e1e2e;border-radius:4px}"
            "QSlider::handle:horizontal{"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #00d4aa,stop:1 #00a88a);border:none;"
            "width:20px;height:20px;margin:-6px 0;border-radius:10px}"
            "QSlider::handle:horizontal:hover{"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,"
            "stop:0 #00ffcc,stop:1 #00d4aa);"
            "width:22px;height:22px;margin:-7px 0;border-radius:11px}"
            "QSlider::sub-page:horizontal{background:#00d4aa;border-radius:4px}")

    def _changed(self, v):
        self.val_lbl.setText(str(v) + self.suffix)
        if self.value_changed_callback:
            self.value_changed_callback(v)

    def value(self):
        return self.slider.value()

    def reset(self):
        self.slider.setValue(self.default_val)


# ============================================================
#  图像显示标签
# ============================================================
class ImageLabel(QLabel):
    def __init__(self, title="", parent=None, keep_aspect=True):
        super().__init__(parent)
        self.title = title
        self._keep_aspect = keep_aspect
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QLabel{background:#0d0d1a;border:2px solid #2a2a3e;"
            "border-radius:10px;"
            "color:#556;font-size:22px;font-weight:bold}")
        self.setText("\n\n  " + title + "  \n\n")
        self._pix = None

    def setImage(self, cv_img):
        if cv_img is None:
            self._pix = None
            self.setText("\n\n  " + self.title + "  \n\n")
            return
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self._pix = QPixmap.fromImage(
            QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888))
        self._scale()

    def _scale(self):
        if self._pix:
            mode = Qt.KeepAspectRatio if self._keep_aspect else Qt.IgnoreAspectRatio
            self.setPixmap(self._pix.scaled(
                self.size(), mode, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._scale()


# ============================================================
#  主窗口
# ============================================================
class ImageProcessor(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像调节工具")
        self.setMinimumSize(1350, 820)
        self.original_img = None

        self._build_ui()
        self._apply_style()

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(25)
        self._timer.timeout.connect(self.process_image)

        self._open_file()

    def _build_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)

        # --- 左侧图像 ---
        sp = QSplitter(Qt.Vertical)
        self.lbl_ori = ImageLabel("原始图像")
        self.lbl_pro = ImageLabel("处理效果")
        self.lbl_hist = ImageLabel("直方图 (RGB)", keep_aspect=False)
        self.lbl_hist.setStyleSheet(
            "QLabel{background:#323232;border:2px solid #505060;"
            "border-radius:10px;color:#888;font-size:20px;font-weight:bold}")
        sp.addWidget(self.lbl_ori)
        sp.addWidget(self.lbl_pro)
        sp.addWidget(self.lbl_hist)
        sp.setSizes([300, 300, 200])
        root.addWidget(sp, 3)

        # --- 右侧面板 ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(480)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel = QWidget()
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(10, 10, 10, 10)
        vbox.setSpacing(10)

        # 基础调节
        g1 = QGroupBox("📐  基础调节")
        v1 = QVBoxLayout(g1)
        self.sl_bri = SliderControl("亮度",   -100, 100, 0)
        self.sl_con = SliderControl("对比度", -50,  50,  0)
        self.sl_sat = SliderControl("饱和度", -100, 100, 0)
        self.sl_gam = SliderControl("Gamma",   20, 200, 100)
        self.sl_gam.slider.valueChanged.connect(self._upd_gam)
        self._upd_gam(100)
        for s in (self.sl_bri, self.sl_con, self.sl_sat, self.sl_gam):
            v1.addWidget(s)
            s.value_changed_callback = self._kick
        vbox.addWidget(g1)

        # 色彩调节
        g2 = QGroupBox("🎨  色彩调节")
        v2 = QVBoxLayout(g2)
        self.sl_hue = SliderControl("色调",    -180, 180, 0)
        self.sl_shadows = SliderControl("阴影", -100, 100, 0)
        self.sl_highlights = SliderControl("高光", -100, 100, 0)
        for s in (self.sl_hue, self.sl_shadows, self.sl_highlights):
            v2.addWidget(s)
            s.value_changed_callback = self._kick
        vbox.addWidget(g2)

        # 滤镜效果
        g3 = QGroupBox("🔬  滤镜效果")
        v3 = QVBoxLayout(g3)
        self.sl_shp = SliderControl("锐化", 0, 100, 0)
        self.sl_mean = SliderControl("均值滤波", 1, 21,  1)  # 新增均值滤波滑块
        self.sl_gau = SliderControl("高斯模糊", 1, 21,  1)
        self.sl_med = SliderControl("中值滤波", 1, 15,  1)
        self.sl_bil = SliderControl("双边滤波", 1, 15,  1)
        for s in (self.sl_shp, self.sl_mean, self.sl_gau, self.sl_med, self.sl_bil):
            v3.addWidget(s)
            s.value_changed_callback = self._kick
        vbox.addWidget(g3)

        # 按钮
        r1 = QHBoxLayout()
        self.btn_open = QPushButton("📂  打开图像")
        self.btn_save = QPushButton("💾  保存结果")
        self.btn_open.clicked.connect(self._open_file)
        self.btn_save.clicked.connect(self._save_result)
        r1.addWidget(self.btn_open)
        r1.addWidget(self.btn_save)
        vbox.addLayout(r1)

        r2 = QHBoxLayout()
        self.btn_rst = QPushButton("🔄  全部重置")
        self.btn_ext = QPushButton("❌  退出")
        self.btn_rst.clicked.connect(self._reset_all)
        self.btn_ext.clicked.connect(self.close)
        r2.addWidget(self.btn_rst)
        r2.addWidget(self.btn_ext)
        vbox.addLayout(r2)

        self.lbl_info = QLabel("尚未加载图像")
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet(
            "color:#aaa;font-size:14px;font-weight:bold;padding:6px")
        vbox.addWidget(self.lbl_info)
        vbox.addStretch()

        scroll.setWidget(panel)
        root.addWidget(scroll)

    def _upd_gam(self, v):
        self.sl_gam.val_lbl.setText(f"{v / 100:.2f}")

    def _kick(self, _=None):
        self._timer.start()

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow{background:#12121e}
            QWidget{background:#12121e;
                font-family:"Microsoft YaHei","Segoe UI",sans-serif}
            QGroupBox{color:#e0e0e0;font-size:16px;font-weight:bold;
                border:1px solid #2a2a3e;border-radius:10px;
                margin-top:14px;padding:20px 10px 10px 10px;
                background:#181828}
            QGroupBox::title{subcontrol-origin:margin;
                subcontrol-position:top left;padding:3px 14px;
                background:#181828;border-radius:5px}
            QPushButton{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2a2a4e,stop:1 #1e1e38);
                color:#d0d0e0;border:1px solid #3a3a5e;
                border-radius:10px;padding:10px 20px;
                font-size:15px;font-weight:bold}
            QPushButton:hover{
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #3a3a6e,stop:1 #2a2a4e);
                border:1px solid #00d4aa;color:#fff}
            QPushButton:pressed{background:#00d4aa;color:#12121e}
            QScrollArea{border:none;background:#12121e}
            QScrollBar:vertical{background:#12121e;width:10px;border-radius:5px}
            QScrollBar::handle:vertical{background:#3a3a5e;border-radius:5px;
                min-height:30px}
            QScrollBar::handle:vertical:hover{background:#00d4aa}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0}
            QSplitter::handle{background:#2a2a3e;height:4px}
        """)

    # --------------------------------------------------
    #  核心
    # --------------------------------------------------
    def _get_result(self):
        img = self.original_img.copy()
        
        # 基础调节
        img = adjust_brightness(img, self.sl_bri.value())
        img = adjust_contrast(img,   self.sl_con.value())
        img = adjust_saturation(img, self.sl_sat.value())
        img = adjust_gamma(img,      self.sl_gam.value() / 100.0)
        img = sharpen_image(img,     self.sl_shp.value())
        
        # 色彩调节
        img = adjust_hue(img,        self.sl_hue.value())
        img = adjust_shadows(img,    self.sl_shadows.value())
        img = adjust_highlights(img, self.sl_highlights.value())
        
        # 滤镜 - 新增均值滤波
        k = self.sl_mean.value()
        if k % 2 == 0:
            k += 1
        img = mean_filter(img, k)
        
        k = self.sl_gau.value()
        if k % 2 == 0:
            k += 1
        img = gaussian_blur(img, k)
        
        k = self.sl_med.value()
        if k % 2 == 0:
            k += 1
        img = median_blur(img, k)
        
        d = self.sl_bil.value()
        if d % 2 == 0:
            d += 1
        img = bilateral_filter(img, d)
        
        return img

    def process_image(self):
        if self.original_img is None:
            return
        result = self._get_result()
        self.lbl_pro.setImage(result)
        hist_img = draw_rgb_histogram(result)
        self.lbl_hist.setImage(hist_img)

    # --------------------------------------------------
    #  文件
    # --------------------------------------------------
    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开图像", "",
            "图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;所有 (*)")
        if not path:
            return
        self.original_img = cv2_imread(path)
        if self.original_img is None:
            self.lbl_info.setText("⚠ 无法读取图像！")
            return
        h, w = self.original_img.shape[:2]
        name = path.replace("\\", "/").split("/")[-1]
        self.lbl_info.setText(f"✅ {name}   {w}×{h}")
        self.lbl_ori.setImage(self.original_img)
        self.process_image()

    def _save_result(self):
        if self.original_img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "result.png",
            "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp);;所有 (*)")
        if not path:
            return
        cv2_imwrite(path, self._get_result())
        name = path.replace("\\", "/").split("/")[-1]
        self.lbl_info.setText(f"💾 已保存: {name}")

    def _reset_all(self):
        for s in (self.sl_bri, self.sl_con, self.sl_sat, self.sl_gam,
                  self.sl_hue, self.sl_shadows, self.sl_highlights,
                  self.sl_shp, self.sl_mean, self.sl_gau, self.sl_med, self.sl_bil):
            s.reset()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 11))
    w = ImageProcessor()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
