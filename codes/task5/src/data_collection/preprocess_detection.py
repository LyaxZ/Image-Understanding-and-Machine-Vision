# -*- coding: utf-8 -*-
"""
使用 YOLOv8 检测车辆并裁剪，保持主体完整
"""

import os
import sys
import cv2
from ultralytics import YOLO
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGED_DIR, PROCESSED_DIR

# 处理后输出目录
OUTPUT_DIR = PROCESSED_DIR
TARGET_SIZE = 224

# 加载 YOLOv8 预训练模型
MODEL_PATH = os.path.join(os.path.dirname(MERGED_DIR), "..", "models", "yolov8n.pt")
model = YOLO(MODEL_PATH)

# 只保留车辆相关类别（COCO 中 car=2, truck=7, bus=5）
VEHICLE_CLASSES = {2, 5, 7}

def detect_and_crop(img_path, save_path):
    try:
        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            return False
        h, w = img.shape[:2]

        # YOLO 检测
        results = model(img, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # 无检测结果，回退到中心裁剪
            return fallback_center_crop(img, save_path)

        # 找出置信度最高的车辆框
        best_box = None
        best_conf = 0.0
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in VEHICLE_CLASSES and conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].cpu().numpy()

        if best_box is None:
            return fallback_center_crop(img, save_path)

        x1, y1, x2, y2 = map(int, best_box)
        # 扩展边界 5%，避免裁得太紧
        pad_w = int((x2 - x1) * 0.05)
        pad_h = int((y2 - y1) * 0.05)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        cropped = img[y1:y2, x1:x2]
        if cropped.size == 0:
            return fallback_center_crop(img, save_path)

        # 等比缩放到目标尺寸（保持宽高比，不足处填黑边）
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cropped)
        pil_img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
        # 创建黑色画布并居中粘贴
        canvas = Image.new('RGB', (TARGET_SIZE, TARGET_SIZE), (0, 0, 0))
        offset = ((TARGET_SIZE - pil_img.width) // 2, (TARGET_SIZE - pil_img.height) // 2)
        canvas.paste(pil_img, offset)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        canvas.save(save_path, 'JPEG', quality=95)
        
        return True

    except Exception as e:
        print(f"检测失败 {img_path}: {e}")
        return False

def fallback_center_crop(img, save_path):
    """中心裁剪回退方案"""
    h, w = img.shape[:2]
    size = min(h, w)
    left = (w - size) // 2
    top = (h - size) // 2
    cropped = img[top:top+size, left:left+size]
    cropped = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE))
    # ✅ 确保目标目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cropped)
    return True

def main():
    if not os.path.exists(MERGED_DIR):
        print(f"错误：merged 目录不存在：{MERGED_DIR}")
        return

    total = 0
    success = 0
    for root, dirs, files in os.walk(MERGED_DIR):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                total += 1
                src_path = os.path.join(root, f)
                rel_path = os.path.relpath(src_path, MERGED_DIR)
                dst_path = os.path.join(OUTPUT_DIR, rel_path)
                dst_path = os.path.splitext(dst_path)[0] + '.jpg'
                if detect_and_crop(src_path, dst_path):
                    success += 1
                    if success % 50 == 0:
                        print(f"已处理 {success}/{total} 张...")

    print(f"\n目标检测辅助裁剪完成！成功 {success}/{total} 张")
    print(f"处理后图片保存在：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()