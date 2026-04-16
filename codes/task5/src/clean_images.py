# -*- coding: utf-8 -*-
"""
安全的图片手动清洗工具（PIL 版，支持中文路径）
按 's' 保留，按 'd' 标记删除，按 'q' 跳过当前类别，按 'r' 重置当前类别的删除标记
"""

import os
import shutil
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

# ========== 配置路径 ==========
BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
# =============================

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

class ImageCleaner:
    def __init__(self, root):
        self.root = root
        self.root.title("图片清洗工具 - 按 S 保留，D 标记删除，Q 跳过类别")
        self.root.geometry("900x700")
        
        self.categories = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
        self.categories.sort()
        self.current_cat_idx = 0
        self.current_img_idx = 0
        self.images = []
        self.kept = 0
        self.deleted = 0
        self.to_delete = set()  # 只记录待删除的文件路径
        
        self.label_info = tk.Label(root, text="", font=("Arial", 12))
        self.label_info.pack(pady=5)
        
        self.canvas = tk.Canvas(root, width=800, height=500, bg="gray")
        self.canvas.pack(pady=10)
        
        self.label_status = tk.Label(root, text="", font=("Arial", 10))
        self.label_status.pack(pady=5)
        
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="保留 (S)", command=self.keep, width=10, bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="删除 (D)", command=self.mark_delete, width=10, bg="lightcoral").pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="跳过类别 (Q)", command=self.skip_category, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="重置标记 (R)", command=self.reset_marks, width=12).pack(side=tk.LEFT, padx=5)
        
        self.root.bind('<s>', lambda e: self.keep())
        self.root.bind('<d>', lambda e: self.mark_delete())
        self.root.bind('<q>', lambda e: self.skip_category())
        self.root.bind('<r>', lambda e: self.reset_marks())
        
        self.load_category()
        
    def load_category(self):
        if self.current_cat_idx >= len(self.categories):
            self.finish()
            return
        cat = self.categories[self.current_cat_idx]
        cat_path = os.path.join(RAW_DIR, cat)
        self.images = [f for f in os.listdir(cat_path) if f.lower().endswith(IMG_EXTENSIONS)]
        self.images.sort()
        self.current_img_idx = 0
        self.kept = 0
        self.deleted = 0
        self.to_delete.clear()
        self.cat = cat
        self.cat_path = cat_path
        self.cleaned_cat_path = os.path.join(CLEANED_DIR, cat)
        os.makedirs(self.cleaned_cat_path, exist_ok=True)
        
        if not self.images:
            print(f"类别 {cat} 无图片，跳过")
            self.current_cat_idx += 1
            self.load_category()
            return
        
        self.show_image()
        
    def show_image(self):
        if self.current_img_idx >= len(self.images):
            # 当前类别结束，执行删除并进入下一类
            self.finish_category()
            return
        
        img_file = self.images[self.current_img_idx]
        img_path = os.path.join(self.cat_path, img_file)
        
        try:
            img = Image.open(img_path)
            # 调整显示大小
            img.thumbnail((780, 480))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(400, 240, image=self.photo, anchor=tk.CENTER)
            
            info_text = f"类别: {self.cat} | 当前: {self.current_img_idx+1}/{len(self.images)} | 已保留: {self.kept} | 待删除: {len(self.to_delete)}"
            self.label_info.config(text=info_text)
            status = "待处理" if img_path not in self.to_delete else "【已标记删除】"
            self.label_status.config(text=f"文件: {img_file}  {status}")
        except Exception as e:
            print(f"无法读取 {img_path}: {e}")
            # 无法读取的图片自动标记删除
            self.to_delete.add(img_path)
            self.current_img_idx += 1
            self.show_image()
    
    def keep(self):
        img_path = os.path.join(self.cat_path, self.images[self.current_img_idx])
        # 复制到 cleaned 目录
        dst = os.path.join(self.cleaned_cat_path, self.images[self.current_img_idx])
        shutil.copy2(img_path, dst)
        self.kept += 1
        self.current_img_idx += 1
        self.show_image()
    
    def mark_delete(self):
        img_path = os.path.join(self.cat_path, self.images[self.current_img_idx])
        self.to_delete.add(img_path)
        self.current_img_idx += 1
        self.show_image()
    
    def reset_marks(self):
        self.to_delete.clear()
        self.current_img_idx = 0
        self.kept = 0
        self.show_image()
    
    def skip_category(self):
        self.finish_category()
    
    def finish_category(self):
        # 执行真正的删除操作
        for path in self.to_delete:
            try:
                os.remove(path)
                self.deleted += 1
            except Exception as e:
                print(f"删除失败 {path}: {e}")
        print(f"✅ {self.cat} 清洗完成：保留 {self.kept} 张，删除 {self.deleted} 张")
        self.current_cat_idx += 1
        self.load_category()
    
    def finish(self):
        self.label_info.config(text="所有类别清洗完毕！")
        self.canvas.delete("all")
        self.canvas.create_text(400, 240, text="完成！可以关闭窗口了。", font=("Arial", 20))
        print("🎉 所有类别清洗完毕！")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCleaner(root)
    root.mainloop()