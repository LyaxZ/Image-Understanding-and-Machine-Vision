"""
主程序：图形界面集成三个任务
"""

import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTabWidget, QTextEdit, QLineEdit, QSpinBox,
                             QFormLayout, QGroupBox, QListWidget, QListWidgetItem,
                             QMessageBox, QProgressBar, QSplitter, QCheckBox, QComboBox)
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

from color_histogram_retrieval import ColorHistogramRetriever, imread_unicode
from hog_feature_extraction import HOGFeatureExtractor
from hog_svm_classifier import HOGSVMClassifier


class ImageLoaderThread(QThread):
    finished = pyqtSignal(str)

    def __init__(self, retriever, folder):
        super().__init__()
        self.retriever = retriever
        self.folder = folder

    def run(self):
        self.retriever.build_database(self.folder)
        self.finished.emit(f"Database loaded: {len(self.retriever.database_paths)} images")


class TrainSVMThread(QThread):
    finished = pyqtSignal(str, float)

    def __init__(self, classifier, data_dir):
        super().__init__()
        self.classifier = classifier
        self.data_dir = data_dir

    def run(self):
        try:
            X, y, class_names = self.classifier.load_data_from_folders(self.data_dir)
            acc = self.classifier.train(X, y)
            self.finished.emit(f"Training done. Accuracy: {acc:.4f}", acc)
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}", 0.0)


class MainWindow(QMainWindow):
    CLASSIFIER_WIN_SIZE = (96, 128)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("图像处理综合实验平台")
        self.setGeometry(100, 100, 1500, 800)

        self.color_retriever = ColorHistogramRetriever(feature_type='auto', hog_win_size=(96,112))
        self.hog_extractor = HOGFeatureExtractor()
        self.classifier = HOGSVMClassifier(win_size=self.CLASSIFIER_WIN_SIZE)

        self.feature_combo = None
        self.hist_query_label = None   # 显示查询图像直方图
        self.hist_result_label = None  # 显示选中结果直方图

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.init_color_tab()
        self.init_hog_tab()
        self.init_svm_tab()

    # ---------- 颜色直方图检索 Tab ----------
    def init_color_tab(self):
        tab = QWidget()
        main_layout = QHBoxLayout(tab)   # 整体水平布局：左侧控制区，右侧显示区

        # ========== 左侧控制区域 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_widget.setFixedWidth(300)

        # 数据库选择
        db_group = QGroupBox("图像数据库")
        db_layout = QVBoxLayout()
        self.db_path_label = QLabel("未选择文件夹")
        self.db_path_label.setWordWrap(True)
        db_select_btn = QPushButton("选择文件夹")
        db_select_btn.clicked.connect(self.select_db_folder)
        db_build_btn = QPushButton("构建特征库")
        db_build_btn.clicked.connect(self.build_database)
        db_layout.addWidget(QLabel("路径:"))
        db_layout.addWidget(self.db_path_label)
        db_layout.addWidget(db_select_btn)
        db_layout.addWidget(db_build_btn)
        db_group.setLayout(db_layout)

        # 特征类型选择
        feature_group = QGroupBox("特征参数")
        feature_layout = QVBoxLayout()
        f1 = QHBoxLayout()
        f1.addWidget(QLabel("特征类型:"))
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(['auto', 'gray', 'lbp', 'hsv', 'hog'])
        self.feature_combo.setCurrentText('auto')
        self.feature_combo.currentTextChanged.connect(self.on_feature_changed)
        f1.addWidget(self.feature_combo)
        feature_layout.addLayout(f1)
        feature_group.setLayout(feature_layout)

        # 查询设置
        query_group = QGroupBox("查询设置")
        query_layout = QVBoxLayout()
        self.query_path_label = QLabel("未选择图像")
        self.query_path_label.setWordWrap(True)
        query_select_btn = QPushButton("选择查询图像")
        query_select_btn.clicked.connect(self.select_query_image)
        q1 = QHBoxLayout()
        q1.addWidget(QLabel("结果数:"))
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(5)
        q1.addWidget(self.top_k_spin)
        search_btn = QPushButton("开始检索")
        search_btn.clicked.connect(self.perform_retrieval)
        query_layout.addWidget(QLabel("图像:"))
        query_layout.addWidget(self.query_path_label)
        query_layout.addWidget(query_select_btn)
        query_layout.addLayout(q1)
        query_layout.addWidget(search_btn)
        query_group.setLayout(query_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        left_layout.addWidget(db_group)
        left_layout.addWidget(feature_group)
        left_layout.addWidget(query_group)
        left_layout.addWidget(self.progress_bar)
        left_layout.addStretch()

        # ========== 右侧显示区域 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # 上半部分：查询预览 + 检索结果列表
        top_splitter = QSplitter(Qt.Horizontal)

        # 查询图像预览
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.addWidget(QLabel("查询图像"))
        self.query_image_label = QLabel()
        self.query_image_label.setFixedSize(250, 250)
        self.query_image_label.setStyleSheet("border:1px solid gray;")
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setText("未选择")
        preview_layout.addWidget(self.query_image_label)

        # 检索结果列表
        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.addWidget(QLabel("检索结果 (点击查看直方图对比)"))
        self.result_list = QListWidget()
        self.result_list.setViewMode(QListWidget.IconMode)
        self.result_list.setIconSize(QSize(120, 120))
        self.result_list.setResizeMode(QListWidget.Adjust)
        self.result_list.itemClicked.connect(self.on_result_clicked)
        result_layout.addWidget(self.result_list)

        top_splitter.addWidget(preview_widget)
        top_splitter.addWidget(result_widget)
        top_splitter.setSizes([250, 500])

        # 下半部分：直方图对比区域
        hist_group = QGroupBox("直方图对比 (左: 查询图像 | 右: 选中结果)")
        hist_layout = QHBoxLayout()
        self.hist_query_label = QLabel()
        self.hist_query_label.setFixedSize(400, 250)
        self.hist_query_label.setStyleSheet("border:1px solid gray; background-color:white;")
        self.hist_query_label.setAlignment(Qt.AlignCenter)
        self.hist_query_label.setText("查询图像直方图")

        self.hist_result_label = QLabel()
        self.hist_result_label.setFixedSize(400, 250)
        self.hist_result_label.setStyleSheet("border:1px solid gray; background-color:white;")
        self.hist_result_label.setAlignment(Qt.AlignCenter)
        self.hist_result_label.setText("点击结果查看直方图")

        hist_layout.addWidget(self.hist_query_label)
        hist_layout.addWidget(self.hist_result_label)
        hist_group.setLayout(hist_layout)

        right_layout.addWidget(top_splitter)
        right_layout.addWidget(hist_group)

        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, stretch=1)

        self.tabs.addTab(tab, "颜色直方图检索")

    def on_feature_changed(self, text):
        self.color_retriever.feature_type = text
        if text == 'hog':
            self.color_retriever.distance_metric = 'euclidean'
        else:
            self.color_retriever.distance_metric = 'chisqr'

        if self.color_retriever.database_features:
            reply = QMessageBox.question(self, "特征类型已更改",
                                         "更改特征类型后，当前特征库将失效。是否立即重建？",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.build_database()
            else:
                self.color_retriever.database_features.clear()
                self.color_retriever.database_paths.clear()
                self.result_list.clear()
                QMessageBox.information(self, "提示", "特征类型已更改，请重新构建数据库。")
        # 更新查询直方图
        if self.query_path_label.text() != "未选择图像":
            self.update_histogram_display(self.query_path_label.text(), is_query=True)

    def select_db_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图像数据库文件夹")
        if folder:
            self.db_path_label.setText(folder)

    def build_database(self):
        if self.feature_combo is not None:
            feature_type = self.feature_combo.currentText()
            self.color_retriever.feature_type = feature_type
            if feature_type == 'hog':
                self.color_retriever.distance_metric = 'euclidean'
            else:
                self.color_retriever.distance_metric = 'chisqr'

        folder = self.db_path_label.text()
        if not folder or folder == "未选择文件夹":
            QMessageBox.warning(self, "警告", "请先选择数据库文件夹")
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.loader_thread = ImageLoaderThread(self.color_retriever, folder)
        self.loader_thread.finished.connect(self.on_db_built)
        self.loader_thread.start()

    def on_db_built(self, msg):
        self.progress_bar.setVisible(False)
        QMessageBox.information(self, "完成", msg)

    def select_query_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择查询图像", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.pgm)")
        if file:
            self.query_path_label.setText(file)
            pixmap = QPixmap(file)
            pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_image_label.setPixmap(pixmap)
            # 显示查询图像的直方图
            self.update_histogram_display(file, is_query=True)

    def update_histogram_display(self, img_path, is_query=True):
        """更新直方图显示，is_query 决定更新左侧还是右侧"""
        img = imread_unicode(img_path)
        if img is None:
            return
        try:
            vis_img = self.color_retriever.get_feature_visualization(img)
            h, w, ch = vis_img.shape
            bytes_per_line = ch * w
            qimg = QImage(vis_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                400, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            if is_query:
                self.hist_query_label.setPixmap(pixmap)
            else:
                self.hist_result_label.setPixmap(pixmap)
        except Exception as e:
            print(f"可视化失败: {e}")

    def perform_retrieval(self):
        query_path = self.query_path_label.text()
        if query_path == "未选择图像" or not os.path.exists(query_path):
            QMessageBox.warning(self, "警告", "请先选择有效的查询图像")
            return

        if not self.color_retriever.database_features:
            QMessageBox.warning(self, "警告", "请先构建特征数据库")
            return

        top_k = self.top_k_spin.value()
        try:
            results = self.color_retriever.retrieve(query_path, top_k)
            self.result_list.clear()
            for path, score in results:
                item = QListWidgetItem()
                icon = QIcon(path)
                item.setIcon(icon)
                item.setText(f"{os.path.basename(path)}\n相似度: {score:.3f}")
                item.setData(Qt.UserRole, path)
                self.result_list.addItem(item)
            # 清空右侧直方图
            self.hist_result_label.clear()
            self.hist_result_label.setText("点击结果查看直方图")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))

    def on_result_clicked(self, item):
        path = item.data(Qt.UserRole)
        # 显示选中图像的直方图
        self.update_histogram_display(path, is_query=False)
        # 可选：显示完整路径
        # QMessageBox.information(self, "图像路径", path)

    # ---------- HOG 特征提取 Tab (保持不变) ----------
    def init_hog_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        param_group = QGroupBox("HOG 参数")
        form = QFormLayout()
        self.cell_size_spin = QSpinBox()
        self.cell_size_spin.setRange(4, 32)
        self.cell_size_spin.setValue(8)
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(1, 4)
        self.block_size_spin.setValue(2)
        self.nbins_spin = QSpinBox()
        self.nbins_spin.setRange(4, 18)
        self.nbins_spin.setValue(9)
        self.signed_cb = QCheckBox()
        form.addRow("Cell Size (pixels):", self.cell_size_spin)
        form.addRow("Block Size (cells):", self.block_size_spin)
        form.addRow("Number of Bins:", self.nbins_spin)
        form.addRow("Signed Gradient:", self.signed_cb)
        param_group.setLayout(form)

        img_layout = QHBoxLayout()
        self.hog_input_label = QLabel("输入图像")
        self.hog_input_label.setFixedSize(300, 300)
        self.hog_input_label.setStyleSheet("border:1px solid gray;")
        self.hog_input_label.setAlignment(Qt.AlignCenter)

        self.hog_output_label = QLabel("HOG 特征向量")
        self.hog_output_label.setFixedSize(400, 300)
        self.hog_output_label.setStyleSheet("border:1px solid gray;")
        self.hog_output_label.setAlignment(Qt.AlignCenter)
        self.hog_output_label.setWordWrap(True)

        img_layout.addWidget(self.hog_input_label)
        img_layout.addWidget(self.hog_output_label)

        btn_layout = QHBoxLayout()
        select_img_btn = QPushButton("选择图像")
        select_img_btn.clicked.connect(self.select_hog_image)
        extract_btn = QPushButton("提取 HOG 特征")
        extract_btn.clicked.connect(self.extract_hog_features)
        btn_layout.addWidget(select_img_btn)
        btn_layout.addWidget(extract_btn)

        layout.addWidget(param_group)
        layout.addLayout(img_layout)
        layout.addLayout(btn_layout)

        self.hog_image_path = None
        self.tabs.addTab(tab, "HOG 特征提取")

    def select_hog_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择图像", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.pgm)")
        if file:
            self.hog_image_path = file
            pixmap = QPixmap(file)
            pixmap = pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.hog_input_label.setPixmap(pixmap)

    def extract_hog_features(self):
        if not self.hog_image_path:
            QMessageBox.warning(self, "警告", "请先选择图像")
            return

        img = imread_unicode(self.hog_image_path)
        if img is None:
            QMessageBox.critical(self, "错误", "无法读取图像")
            return

        cell_size = (self.cell_size_spin.value(), self.cell_size_spin.value())
        block_size = (self.block_size_spin.value(), self.block_size_spin.value())
        nbins = self.nbins_spin.value()
        signed = self.signed_cb.isChecked()

        extractor = HOGFeatureExtractor(cell_size=cell_size,
                                        block_size=block_size,
                                        nbins=nbins,
                                        signed_gradient=signed)
        try:
            features = extractor.compute_hog(img)
            feat_str = f"特征向量长度: {len(features)}\n前20个值:\n"
            feat_str += np.array2string(features[:20], precision=4, separator=', ')
            self.hog_output_label.setText(feat_str)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"提取失败: {str(e)}")

    # ---------- HOG+SVM 分类 Tab (保持不变) ----------
    def init_svm_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        train_group = QGroupBox("训练模型")
        train_layout = QVBoxLayout()

        h1 = QHBoxLayout()
        self.train_data_label = QLabel("未选择数据集文件夹")
        select_data_btn = QPushButton("选择数据集文件夹")
        select_data_btn.clicked.connect(self.select_svm_data_folder)
        h1.addWidget(QLabel("数据集:"))
        h1.addWidget(self.train_data_label)
        h1.addWidget(select_data_btn)

        train_btn = QPushButton("开始训练")
        train_btn.clicked.connect(self.train_svm)
        self.svm_progress = QProgressBar()
        self.svm_progress.setVisible(False)

        train_layout.addLayout(h1)
        train_layout.addWidget(train_btn)
        train_layout.addWidget(self.svm_progress)
        train_group.setLayout(train_layout)

        predict_group = QGroupBox("图像分类预测")
        predict_layout = QVBoxLayout()

        h2 = QHBoxLayout()
        self.predict_img_label = QLabel()
        self.predict_img_label.setFixedSize(200, 200)
        self.predict_img_label.setStyleSheet("border:1px solid gray;")
        self.predict_img_label.setAlignment(Qt.AlignCenter)
        self.predict_img_label.setText("待预测图像")

        self.predict_result_label = QLabel("预测结果: ")
        h2.addWidget(self.predict_img_label)
        h2.addWidget(self.predict_result_label)

        h3 = QHBoxLayout()
        select_predict_btn = QPushButton("选择图像")
        select_predict_btn.clicked.connect(self.select_predict_image)
        predict_btn = QPushButton("预测")
        predict_btn.clicked.connect(self.predict_image)
        load_model_btn = QPushButton("加载模型")
        load_model_btn.clicked.connect(self.load_svm_model)
        save_model_btn = QPushButton("保存模型")
        save_model_btn.clicked.connect(self.save_svm_model)

        h3.addWidget(select_predict_btn)
        h3.addWidget(predict_btn)
        h3.addWidget(load_model_btn)
        h3.addWidget(save_model_btn)

        predict_layout.addLayout(h2)
        predict_layout.addLayout(h3)
        predict_group.setLayout(predict_layout)

        layout.addWidget(train_group)
        layout.addWidget(predict_group)

        self.predict_image_path = None
        self.svm_data_dir = None

        self.tabs.addTab(tab, "HOG+SVM 分类")

    def select_svm_data_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "选择数据集根目录（每个子文件夹为一个类别）")
        if folder:
            self.svm_data_dir = folder
            self.train_data_label.setText(folder)

    def train_svm(self):
        if not self.svm_data_dir:
            QMessageBox.warning(self, "警告", "请先选择数据集文件夹")
            return

        self.svm_progress.setVisible(True)
        self.svm_progress.setRange(0, 0)
        self.train_thread = TrainSVMThread(self.classifier, self.svm_data_dir)
        self.train_thread.finished.connect(self.on_svm_trained)
        self.train_thread.start()

    def on_svm_trained(self, msg, acc):
        self.svm_progress.setVisible(False)
        if "Accuracy:" in msg:
            parts = msg.split("Accuracy:")
            if len(parts) == 2:
                try:
                    acc_value = float(parts[1].strip())
                    msg = f"{parts[0]}Accuracy: {acc_value*100:.2f}%"
                except:
                    pass
        QMessageBox.information(self, "训练完成", msg)

    def select_predict_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择待预测图像", "",
                                              "Images (*.png *.jpg *.jpeg *.bmp *.pgm)")
        if file:
            self.predict_image_path = file
            pixmap = QPixmap(file)
            pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.predict_img_label.setPixmap(pixmap)

    def predict_image(self):
        if not self.predict_image_path:
            QMessageBox.warning(self, "警告", "请先选择待预测图像")
            return
        if not self.classifier.class_names:
            QMessageBox.warning(self, "警告", "请先训练或加载模型")
            return

        img = imread_unicode(self.predict_image_path)
        if img is None:
            QMessageBox.critical(self, "错误", "无法读取图像")
            return

        try:
            idx, name, conf = self.classifier.predict(img)
            self.predict_result_label.setText(f"预测类别: {name}\n置信度: {conf:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败: {str(e)}")

    def save_svm_model(self):
        if not self.classifier.class_names:
            QMessageBox.warning(self, "警告", "没有可保存的模型")
            return
        file, _ = QFileDialog.getSaveFileName(self, "保存模型", "", "Model Files (*.pkl)")
        if file:
            self.classifier.save_model(file)
            QMessageBox.information(self, "成功", "模型已保存")

    def load_svm_model(self):
        file, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "Model Files (*.pkl)")
        if file:
            try:
                self.classifier.load_model(file)
                QMessageBox.information(self, "成功", f"模型已加载，类别: {self.classifier.class_names}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载失败: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())