# -*- coding: utf-8 -*-
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class CarBrandDataset(Dataset):
    def __init__(self, csv_path, img_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['path'])
        label = row['label']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # 损坏图片用全黑图代替，避免中断
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, label