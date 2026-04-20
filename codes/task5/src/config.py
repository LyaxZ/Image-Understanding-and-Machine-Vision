# -*- coding: utf-8 -*-
"""
全局配置文件
请在下方修改 PROJECT_ROOT 为你的项目根目录（task5 所在路径）
"""

# ========== 请修改这里的路径 ==========
# D:\College\大三下\图像理解与机器视觉\Expe\codes\task5
# D:\DeepLearning\Image-Understanding-and-Machine-Vision\codes\task5
PROJECT_ROOT = r"D:\DeepLearning\Image-Understanding-and-Machine-Vision\codes\task5"
# ====================================

import os

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_NEW_DIR = os.path.join(DATA_DIR, "raw_new")
MERGED_DIR = os.path.join(DATA_DIR, "merged")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_224")
AUGMENTED_DIR = os.path.join(DATA_DIR, "augmented")

# CSV 文件路径
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")

# 模型与日志目录
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# 确保关键目录存在
for d in [DATA_DIR, RAW_DIR, RAW_NEW_DIR, MERGED_DIR, AUGMENTED_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# 类别相关（可根据实际情况调整）
# 原始6类映射（用于合并脚本）
BRAND_MAP_6 = {
    "大众朗逸": "大众", "大众速腾": "大众", "大众帕萨特": "大众",
    "丰田卡罗拉": "丰田", "丰田凯美瑞": "丰田",
    "本田思域": "本田", "本田雅阁": "本田",
    "宝马3系": "宝马", "宝马5系": "宝马",
    "奔驰C级": "奔驰", "奔驰E级": "奔驰", "奔驰S级": "奔驰",
    "比亚迪秦": "比亚迪", "比亚迪汉": "比亚迪"
}

# 新品牌多车型爬虫配置
NEW_BRAND_MODELS = {
    "奥迪": ["奥迪 A4L", "奥迪 A6L", "奥迪 Q5L"],
    "特斯拉": ["特斯拉 Model 3", "特斯拉 Model Y"],
    "日产": ["日产 轩逸", "日产 天籁", "日产 逍客"],
    "凯迪拉克": ["凯迪拉克 CT5", "凯迪拉克 XT5"],
    "蔚来": ["蔚来 ET5", "蔚来 ES6", "蔚来 ET7"],
    "理想": ["理想 L7", "理想 L8", "理想 L9"],
    "问界": ["问界 M5", "问界 M7"]
}