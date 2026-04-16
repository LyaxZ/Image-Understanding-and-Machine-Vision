import os

BASE_DIR = r"D:\College\大三下\图像理解与机器视觉\Expe\codes\task5"
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")

print("清洗后各类别图片数量：")
total = 0
for folder in sorted(os.listdir(CLEANED_DIR)):
    folder_path = os.path.join(CLEANED_DIR, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.png','.jpeg'))])
        print(f"{folder}: {count} 张")
        total += count
print(f"\n总计：{total} 张")