import os
aug_dir = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5\data\augmented"
for cls in os.listdir(aug_dir):
    path = os.path.join(aug_dir, cls)
    if os.path.isdir(path):
        cnt = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        print(f"{cls}: {cnt} 张")