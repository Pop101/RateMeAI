"""ConvNeXt-as-frozen-backbone + ``MLPRegressionHead`` (Lightning)."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

from modules.heads import MLPRegressionHead
from mltk import AbstractModel, auto_device

device = auto_device()


class ConvnextModel(AbstractModel):
    def __init__(self, lr: float = 0.001, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.save_hyperparameters({"lr": lr})

        weights = ConvNeXt_Base_Weights.DEFAULT
        backbone = convnext_base(weights=weights)
        if not (
            hasattr(backbone, 'classifier')
            and isinstance(backbone.classifier[-1], nn.Linear)
        ):
            raise ValueError(
                "ConvNeXt classifier shape changed; expected last layer to be nn.Linear"
            )
        feat_dim = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self._backbone = backbone.to(device=device, dtype=dtype)

        self.head = MLPRegressionHead(feat_dim, out_dim=1)

        prep = weights.transforms()
        self.normalize = transforms.Normalize(mean=tuple(prep.mean), std=tuple(prep.std))
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
