# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CarBrandDataset
from improved_cnn import ImprovedCNN

BASE_DIR = r"D:\DeepLearning\Image-Understanding-and-Machine-Vision\codes\task5"
DATA_DIR = os.path.join(BASE_DIR, "data", "augmented")
TEST_CSV = os.path.join(BASE_DIR, "data", "test.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "improved_cnn_best.pth")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CarBrandDataset(TEST_CSV, DATA_DIR, transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

model = ImprovedCNN(num_classes=6).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

correct = total = 0
with torch.no_grad():
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct / total:.4f}")