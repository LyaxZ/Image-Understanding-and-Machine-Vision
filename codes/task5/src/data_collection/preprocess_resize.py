# -*- coding: utf-8 -*-
"""
离线预处理脚本：将 merged/ 中的所有图片统一转换为 224x224（保持比例，中心裁剪）
输出到 data/processed_224/，文件夹结构与 merged 一致。
"""

import os
import sys
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGED_DIR, DATA_DIR, PROCESSED_DIR

def resize_and_crop(img, target_size=224):
    """
    保持宽高比缩放，使短边等于 target_size，然后中心裁剪 target_size x target_size
    """
    # 计算缩放比例（使短边为 target_size）
    w, h = img.size
    if w < h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 中心裁剪
    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2
    right = left + target_size
    bottom = top + target_size
    img = img.crop((left, top, right, bottom))
    return img

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    classes = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]

    for cls in classes:
        cls_src = os.path.join(MERGED_DIR, cls)
        cls_dst = os.path.join(PROCESSED_DIR, cls)
        os.makedirs(cls_dst, exist_ok=True)

        for fname in os.listdir(cls_src):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            src_path = os.path.join(cls_src, fname)
            dst_path = os.path.join(cls_dst, fname)
            try:
                img = Image.open(src_path).convert('RGB')
                img = resize_and_crop(img, 224)
                img.save(dst_path)
            except Exception as e:
                print(f"处理失败 {src_path}: {e}")

        print(f"✅ {cls} 预处理完成")

    print(f"\n所有图片已保存至 {PROCESSED_DIR}")

if __name__ == "__main__":
    main()