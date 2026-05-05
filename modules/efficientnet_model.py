"""EfficientNet-as-frozen-backbone + ``MLPRegressionHead`` (Lightning).

Lifecycle (train/val/checkpoint/device-move) comes from ``AbstractModel`` ->
``LightningModule``. The frozen backbone is intentionally NOT a registered
child module: weights don't change, don't need saving, and the upstream
EfficientNet weights are re-fetched on construction.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

from modules.heads import MLPRegressionHead
from mltk import AbstractModel, auto_device

device = auto_device()

BASE_MODEL = 'efficientnet-b0'

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class EfficientNetModel(AbstractModel):
    """B0: 224×224 / B1: 240×240 / B2: 260×260 / B3: 300×300 /
    B4: 380×380 / B5: 456×456 / B6: 528×528 / B7: 600×600. Wrong
    size → NaN loss."""

    def __init__(
        self,
        model_type: str = BASE_MODEL,
        lr: float = 0.001,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.save_hyperparameters({"model_type": model_type, "lr": lr})
        self.model_type = model_type

        backbone = EfficientNet.from_pretrained(model_type)
        feat_dim = backbone._fc.in_features
        backbone._fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self._backbone = backbone.to(device=device, dtype=dtype)

        self.head = MLPRegressionHead(feat_dim, out_dim=1)
        self.normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, cooldown=3, min_lr=1e-6
        )
        self.scheduler.interval = "epoch"
        self.scheduler.monitor = "val_loss"

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_backbone(image)).squeeze(-1)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)

    def forward_backbone(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._backbone(image)
