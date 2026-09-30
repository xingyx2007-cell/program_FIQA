"""MobileNetV3-Small quality regressors."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class MultiScaleMobileNetV3Small(nn.Module):
    """Add pooled shallow and middle features to the pretrained deep embedding."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier
        self.shallow_projection = nn.Linear(24, 16)
        self.mid_projection = nn.Linear(40, 16)
        self.classifier[3] = nn.Linear(self.classifier[3].in_features + 32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = mid = None
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index == 2:
                shallow = F.adaptive_avg_pool2d(x, 1).flatten(1)
            elif index == 6:
                mid = F.adaptive_avg_pool2d(x, 1).flatten(1)
        deep = self.classifier[:3](self.avgpool(x).flatten(1))
        fused = torch.cat((deep, self.shallow_projection(shallow),
                           self.mid_projection(mid)), dim=1)
        return self.classifier[3](fused)


def build_model(use_pretrained: bool = False, pretrained_weights_path: str | None = None,
                multi_scale: bool = False) -> nn.Module:
    if pretrained_weights_path is not None:
        raise ValueError("Only torchvision's official IMAGENET1K_V1 weights are permitted")
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
    model = mobilenet_v3_small(weights=weights)
    if multi_scale:
        return MultiScaleMobileNetV3Small(model)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model
