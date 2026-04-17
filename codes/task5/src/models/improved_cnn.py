# -*- coding: utf-8 -*-
import torch.nn as nn

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=13):   # 默认13类
        super(ImprovedCNN, self).__init__()
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1_1   = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2   = nn.BatchNorm2d(64)
        self.pool1   = nn.MaxPool2d(2, 2)
        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1   = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2   = nn.BatchNorm2d(128)
        self.pool2   = nn.MaxPool2d(2, 2)
        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(256)
        self.pool3   = nn.MaxPool2d(2, 2)
        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4_1   = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4_2   = nn.BatchNorm2d(512)
        self.pool4   = nn.MaxPool2d(2, 2)
        # Classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.relu(self.bn4_1(self.conv4_1(x)))
        x = self.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x