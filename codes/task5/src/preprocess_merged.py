# -*- coding: utf-8 -*-
"""
将 merged 中的原始图片预处理为统一 224x224 的正方形图像
保持宽高比、中心裁剪，无拉伸变形
"""

import os
from PIL import Image

BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
MERGED_DIR = os.path.join(BASE_DIR, "data", "merged")
PREPROCESSED_DIR = os.path.join(BASE_DIR, "data", "preprocessed")

TARGET_SIZE = 224
RESIZE_SHORT = 256

def preprocess_image(img_path, save_path):
    """对单张图片进行保持比例的缩放+中心裁剪，保存"""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"  跳过损坏文件: {img_path}")
        return

    # 1. 保持宽高比，将短边缩放到 RESIZE_SHORT
    w, h = img.size
    if w < h:
        new_w = RESIZE_SHORT
        new_h = int(h * RESIZE_SHORT / w)
    else:
        new_h = RESIZE_SHORT
        new_w = int(w * RESIZE_SHORT / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # 2. 中心裁剪 TARGET_SIZE x TARGET_SIZE
    left = (new_w - TARGET_SIZE) // 2
    top = (new_h - TARGET_SIZE) // 2
    right = left + TARGET_SIZE
    bottom = top + TARGET_SIZE
    img = img.crop((left, top, right, bottom))

    # 保存
    img.save(save_path)

def main():
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    classes = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]

    for cls in classes:
        src_cls = os.path.join(MERGED_DIR, cls)
        dst_cls = os.path.join(PREPROCESSED_DIR, cls)
        os.makedirs(dst_cls, exist_ok=True)

        images = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        print(f"处理类别 {cls}: {len(images)} 张")

        for img_file in images:
            src_path = os.path.join(src_cls, img_file)
            dst_path = os.path.join(dst_cls, img_file)
            preprocess_image(src_path, dst_path)

    print("\n✅ 所有图片预处理完成！")
    print(f"预处理后图片保存至: {PREPROCESSED_DIR}")

if __name__ == "__main__":
    main()