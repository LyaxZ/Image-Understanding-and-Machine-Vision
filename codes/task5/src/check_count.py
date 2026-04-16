# -*- coding: utf-8 -*-
"""
统计各类别图片数量（绝对路径版）
"""

import os

# 请确认路径与 crawler.py 中的 BASE_DIR 一致
BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

print("各类别图片数量统计：")
print(f"检查路径：{RAW_DATA_DIR}\n")

if not os.path.exists(RAW_DATA_DIR):
    print(f"❌ 路径不存在：{RAW_DATA_DIR}")
else:
    total = 0
    for folder in sorted(os.listdir(RAW_DATA_DIR)):
        folder_path = os.path.join(RAW_DATA_DIR, folder)
        if os.path.isdir(folder_path):
            count = len([f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            print(f"{folder}: {count} 张")
            total += count
    print(f"\n总计：{total} 张图片")
