# -*- coding: utf-8 -*-
"""
按品牌合并清洗后的图片
将 cleaned 目录下的车型文件夹合并到 merged 目录的品牌文件夹中
"""

import os
import shutil

BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
MERGED_DIR = os.path.join(BASE_DIR, "data", "merged")

# 车型到品牌的映射
BRAND_MAP = {
    "大众朗逸": "大众",
    "大众速腾": "大众",
    "大众帕萨特": "大众",
    "丰田卡罗拉": "丰田",
    "丰田凯美瑞": "丰田",
    "本田思域": "本田",
    "本田雅阁": "本田",
    "宝马3系": "宝马",
    "宝马5系": "宝马",
    "奔驰C级": "奔驰",
    "奔驰E级": "奔驰",
    "奔驰S级": "奔驰",
    "比亚迪秦": "比亚迪",
    "比亚迪汉": "比亚迪"
}

def merge():
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    for car_model, brand in BRAND_MAP.items():
        # 源文件夹（清洗后的车型文件夹）
        src_folder = os.path.join(CLEANED_DIR, car_model.replace(" ", "_"))
        if not os.path.exists(src_folder):
            print(f"⚠️ 源文件夹不存在：{src_folder}")
            continue
        
        # 目标文件夹（merged/品牌名）
        dst_folder = os.path.join(MERGED_DIR, brand)
        os.makedirs(dst_folder, exist_ok=True)
        
        # 复制所有图片，加上车型前缀避免重名
        for img_file in os.listdir(src_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                src_path = os.path.join(src_folder, img_file)
                # 新文件名：车型_原文件名
                new_name = f"{car_model.replace(' ', '_')}_{img_file}"
                dst_path = os.path.join(dst_folder, new_name)
                shutil.copy2(src_path, dst_path)
        
        print(f"✅ {car_model} → {brand} 完成")
    
    # 统计合并后各类数量
    print("\n合并后各类图片数量：")
    for brand in sorted(os.listdir(MERGED_DIR)):
        brand_path = os.path.join(MERGED_DIR, brand)
        if os.path.isdir(brand_path):
            count = len([f for f in os.listdir(brand_path) if f.lower().endswith(('.jpg','.png','.jpeg'))])
            print(f"{brand}: {count} 张")

if __name__ == "__main__":
    merge()