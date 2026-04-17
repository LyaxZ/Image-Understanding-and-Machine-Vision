# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from config import AUGMENTED_DIR, TEST_CSV, MODELS_DIR
from utils.dataset import CarBrandDataset
from models.improved_cnn import ImprovedCNN

NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(MODELS_DIR, f"improved_cnn_{NUM_CLASSES}class_best.pth")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = CarBrandDataset(TEST_CSV, AUGMENTED_DIR, transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

model = ImprovedCNN(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

correct = total = 0
with torch.no_grad():
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")