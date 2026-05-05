"""ViT-B/16-as-frozen-backbone + ``MLPRegressionHead`` (Lightning).

``input_size`` other than 224 still works — the positional embedding is
resampled in ``_modify_model_for_input_size``. We do this *before* freezing
so the resized embedding is the one that ends up ``requires_grad=False``.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import ViT_B_16_Weights, vit_b_16

from modules.heads import MLPRegressionHead
from mltk import AbstractModel, auto_device

device = auto_device()


class VisionTransformerModel(AbstractModel):
    def __init__(
        self,
        input_size: int = 224,
        lr: float = 0.001,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.save_hyperparameters({"input_size": input_size, "lr": lr})
        self.input_size = input_size

        weights = ViT_B_16_Weights.DEFAULT
        backbone = vit_b_16(weights=weights)
        if input_size != 224:
            self._modify_model_for_input_size(backbone, input_size)

        feat_dim = backbone.hidden_dim
        backbone.heads = nn.Identity()
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

    @staticmethod
    def _modify_model_for_input_size(model, input_size: int) -> None:
        model.image_size = input_size
        orig_pos_embed = model.encoder.pos_embedding
        patch_size = model.patch_size
        orig_size = 224 // patch_size
        new_size = input_size // patch_size

        cls_pos_embed = orig_pos_embed[:, 0:1]
        pos_embed = orig_pos_embed[:, 1:]
        dim = pos_embed.shape[-1]
        pos_embed = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        new_pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
        model.encoder.pos_embedding = nn.Parameter(new_pos_embed)
        model.seq_length = new_pos_embed.shape[1]
