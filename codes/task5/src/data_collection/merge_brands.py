# -*- coding: utf-8 -*-
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DIR, MERGED_DIR, BRAND_MAP_6

def main():
    for folder in os.listdir(RAW_DIR):
        folder_path = os.path.join(RAW_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        # 如果是车型文件夹（在映射中），则归到品牌下；否则直接使用文件夹名（如新品牌）
        brand = BRAND_MAP_6.get(folder, folder)
        dest_dir = os.path.join(MERGED_DIR, brand)
        os.makedirs(dest_dir, exist_ok=True)
        for f in os.listdir(folder_path):
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                shutil.copy2(os.path.join(folder_path, f), os.path.join(dest_dir, f))
        print(f"{folder} -> {brand}")
    print("合并完成！")

if __name__ == "__main__":
    main()