# -*- coding: utf-8 -*-
import os
import sys
import shutil
from PIL import Image, ImageTk
import tkinter as tk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_NEW_DIR, MERGED_DIR  # 可根据需要修改为 RAW_DIR

# 设置要清洗的源目录（可手动改为 RAW_DIR 或 RAW_NEW_DIR）
SOURCE_DIR = RAW_NEW_DIR
DEST_DIR = MERGED_DIR

class ImageCleaner:
    def __init__(self, root):
        self.root = root
        self.root.title("清洗工具 - S保留 D删除 Q跳过")
        self.categories = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
        self.cat_idx = 0
        self.img_idx = 0
        self.images = []
        self.to_delete = set()
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()
        self.label = tk.Label(root, text="", font=("Arial", 12))
        self.label.pack()
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="保留 (S)", command=self.keep).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="删除 (D)", command=self.delete).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="跳过类别 (Q)", command=self.finish_category).pack(side=tk.LEFT)
        self.root.bind('<s>', lambda e: self.keep())
        self.root.bind('<d>', lambda e: self.delete())
        self.root.bind('<q>', lambda e: self.finish_category())
        self.load_category()

    def load_category(self):
        if self.cat_idx >= len(self.categories):
            self.root.destroy()
            return
        cat = self.categories[self.cat_idx]
        self.cat_path = os.path.join(SOURCE_DIR, cat)
        self.clean_path = os.path.join(DEST_DIR, cat)
        os.makedirs(self.clean_path, exist_ok=True)
        self.images = [f for f in os.listdir(self.cat_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        self.img_idx = 0
        self.to_delete.clear()
        self.show_image()

    def show_image(self):
        if self.img_idx >= len(self.images):
            self.finish_category()
            return
        img_path = os.path.join(self.cat_path, self.images[self.img_idx])
        try:
            img = Image.open(img_path)
            img.thumbnail((800, 600))
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.create_image(400, 300, image=self.photo)
            self.label.config(text=f"{self.categories[self.cat_idx]}  {self.img_idx+1}/{len(self.images)}")
        except:
            self.to_delete.add(img_path)
            self.img_idx += 1
            self.show_image()

    def keep(self):
        src = os.path.join(self.cat_path, self.images[self.img_idx])
        dst = os.path.join(self.clean_path, self.images[self.img_idx])
        shutil.copy2(src, dst)
        self.img_idx += 1
        self.show_image()

    def delete(self):
        self.to_delete.add(os.path.join(self.cat_path, self.images[self.img_idx]))
        self.img_idx += 1
        self.show_image()

    def finish_category(self):
        for p in self.to_delete:
            if os.path.exists(p):
                os.remove(p)
        print(f"{self.categories[self.cat_idx]} 完成，保留 {len(self.images)-len(self.to_delete)} 张")
        self.cat_idx += 1
        self.load_category()

def main():
    root = tk.Tk()
    ImageCleaner(root)
    root.mainloop()

if __name__ == "__main__":
    main()