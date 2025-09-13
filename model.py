# model.py
import torch
import torch.nn as nn
import torchvision.models as models


class VolleyballBaseline1(nn.Module):
    def __init__(self, num_group_classes, backbone_name='resnet18'):
        super().__init__()
        # Load pre-trained backbone
        self.backbone = getattr(models, backbone_name)(weights="IMAGENET1K_V1")

        # Replace final FC layer for group activity classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_group_classes)

        # Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last conv block + classifier
        if 'resnet' in backbone_name:
            for name, param in self.backbone.named_parameters():
                if name.startswith("layer4") or name.startswith("fc"):
                    param.requires_grad = True
        else:
            # Other backbones → only unfreeze classifier
            for param in self.backbone.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        # x: (B, T, C, H, W) with T=1 for single-frame mode
        B, T, C, H, W = x.shape
        assert T == 1, f"Expected single frame (T=1), got T={T}"
        x = x.squeeze(1)  # (B, C, H, W)
        return self.backbone(x)
