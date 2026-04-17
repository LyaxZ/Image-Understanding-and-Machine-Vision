# -*- coding: utf-8 -*-
import os
import sys
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGED_DIR, AUGMENTED_DIR

TARGET_PER_CLASS = 700

def augment(img):
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    img = img.rotate(random.uniform(-10, 10))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1)))
    return img

def main():
    for cls in os.listdir(MERGED_DIR):
        cls_path = os.path.join(MERGED_DIR, cls)
        if not os.path.isdir(cls_path):
            continue
        out_path = os.path.join(AUGMENTED_DIR, cls)
        os.makedirs(out_path, exist_ok=True)
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        # 复制原图
        for f in images:
            img = Image.open(os.path.join(cls_path, f)).convert('RGB')
            img.save(os.path.join(out_path, f))
        need = TARGET_PER_CLASS - len(images)
        if need <= 0:
            print(f"{cls} 已达标 ({len(images)}张)")
            continue
        generated = 0
        while generated < need:
            for f in images:
                if generated >= need:
                    break
                img = Image.open(os.path.join(cls_path, f)).convert('RGB')
                aug_img = augment(img)
                new_name = f"{os.path.splitext(f)[0]}_aug{generated:04d}.jpg"
                aug_img.save(os.path.join(out_path, new_name))
                generated += 1
        print(f"{cls}: {len(images)} -> {len(os.listdir(out_path))}")
    print("增强完成！")

if __name__ == "__main__":
    main()