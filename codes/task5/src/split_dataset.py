# -*- coding: utf-8 -*-
"""
将 augmented 数据集按 7:2:1 划分为 train/val/test，生成 CSV
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def main():
    data = []
    class_names = sorted([d for d in os.listdir(AUGMENTED_DIR) if os.path.isdir(os.path.join(AUGMENTED_DIR, d))])
    label_map = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(AUGMENTED_DIR, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg','.png','.jpeg')):
                rel_path = os.path.join(class_name, img_file)
                data.append([rel_path, label_map[class_name]])

    df = pd.DataFrame(data, columns=["path", "label"])
    train, temp = train_test_split(df, test_size=1-TRAIN_RATIO, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), stratify=temp["label"], random_state=42)

    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"类别映射：{label_map}")
    print(f"训练集：{len(train)} 张")
    print(f"验证集：{len(val)} 张")
    print(f"测试集：{len(test)} 张")
    print("✅ CSV 文件已保存到 data 目录。")

if __name__ == "__main__":
    main()