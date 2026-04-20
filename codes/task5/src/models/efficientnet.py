# -*- coding: utf-8 -*-
import timm
import torch.nn as nn

class EfficientNetTransfer(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(EfficientNetTransfer, self).__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained)
        in_features = self.backbone.classifier.in_features
        # 替换分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)