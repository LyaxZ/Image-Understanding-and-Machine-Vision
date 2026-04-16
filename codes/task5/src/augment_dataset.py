# -*- coding: utf-8 -*-
"""
数据增强脚本 (PIL版，完美支持中文路径 + 修复RGBA问题)
"""

import os
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
MERGED_DIR = os.path.join(BASE_DIR, "data", "merged")
AUGMENTED_DIR = os.path.join(BASE_DIR, "data", "augmented")

TARGET_PER_CLASS = 700

def pil_augment(image):
    """对RGB模式的PIL图像应用随机增强，返回增强后的RGB图像"""
    # 随机水平翻转
    if random.random() < 0.5:
        image = ImageOps.mirror(image)
    
    # 随机旋转 -10 到 10 度
    angle = random.uniform(-10, 10)
    image = image.rotate(angle, expand=False, fillcolor=(0,0,0))
    
    # 随机亮度调整
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 随机对比度调整
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    
    # 随机轻微模糊
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.0)))
    
    return image

def save_image(image, path):
    """根据扩展名自动选择保存格式，强制转换为RGB（JPEG不支持RGBA）"""
    ext = os.path.splitext(path)[1].lower()
    # 统一转为RGB模式（丢弃透明通道）
    if image.mode in ('RGBA', 'LA', 'P'):
        image = image.convert('RGB')
    # 如果是JPEG/JPG且图像模式不是RGB，转换一次
    if ext in ('.jpg', '.jpeg') and image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(path)

def augment_class(class_name):
    src_folder = os.path.join(MERGED_DIR, class_name)
    dst_folder = os.path.join(AUGMENTED_DIR, class_name)
    os.makedirs(dst_folder, exist_ok=True)

    images = [f for f in os.listdir(src_folder) 
              if f.lower().endswith(('.jpg','.jpeg','.png'))]
    current_count = len(images)
    needed = TARGET_PER_CLASS - current_count

    print(f"{class_name}: 原始 {current_count} 张", end="")

    if needed <= 0:
        print(" → 已达标，仅复制原图")
        for img in images:
            src = os.path.join(src_folder, img)
            dst = os.path.join(dst_folder, img)
            img_pil = Image.open(src)
            save_image(img_pil, dst)
        return

    print(f" → 需生成 {needed} 张增强图片")

    # 复制原图
    for img in images:
        src = os.path.join(src_folder, img)
        dst = os.path.join(dst_folder, img)
        img_pil = Image.open(src)
        save_image(img_pil, dst)

    # 生成增强图片
    generated = 0
    while generated < needed:
        for img_file in images:
            if generated >= needed:
                break
            src_path = os.path.join(src_folder, img_file)
            try:
                img = Image.open(src_path)
                # 关键：转为RGB模式
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
            except Exception as e:
                print(f"\n  警告：无法读取 {img_file}，跳过")
                continue

            aug_img = pil_augment(img)
            name, ext = os.path.splitext(img_file)
            new_name = f"{name}_aug{generated:04d}{ext}"
            dst_path = os.path.join(dst_folder, new_name)
            save_image(aug_img, dst_path)
            generated += 1

    final_count = len(os.listdir(dst_folder))
    print(f"  ✅ 完成，最终 {final_count} 张")

def main():
    os.makedirs(AUGMENTED_DIR, exist_ok=True)
    classes = [d for d in os.listdir(MERGED_DIR) if os.path.isdir(os.path.join(MERGED_DIR, d))]
    for cls in sorted(classes):
        augment_class(cls)
    print("\n🎉 所有类别增强完毕！")

if __name__ == "__main__":
    main()