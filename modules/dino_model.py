"""DINOv3 backbone + ``MLPRegressionHead`` (Lightning).

The frozen DINOv3 backbone is owned by the local ``BackboneSpec`` (see
``modules/backbones.py``) and is intentionally NOT a registered child
module: its weights don't change, don't need saving, and ``load_backbone``
re-fetches them on construction. ``forward_backbone`` runs it under
``no_grad`` so the autograd graph stays confined to the head.

Note: DINOv3 weights are gated on HuggingFace. To download you must:
  1. Accept the license at https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
  2. Authenticate locally with ``huggingface-cli login`` (or set
     ``HF_TOKEN`` in the environment).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from modules.backbones import load_backbone
from modules.heads import MLPRegressionHead
from mltk import AbstractModel, auto_device

device = auto_device()

# DINOv3 ViT-S/16 — small enough to feature-extract on CPU in reasonable
# time, ~22M params, 384-dim CLS embedding. Swap to dinov3-vitb16-... or
# dinov3-vitl16-... for stronger features at higher cost.
DEFAULT_DINO_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


class DinoVisionAnalysisModel(AbstractModel):
    def __init__(
        self,
        lr: float = 0.001,
        dino_model_name: str = DEFAULT_DINO_MODEL,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.save_hyperparameters({"lr": lr, "dino_model_name": dino_model_name})
        self.dino_model_name = dino_model_name

        self.spec = load_backbone(
            hf_model_id=dino_model_name,
            device=device,
            dtype=dtype,
        )

        self.head = MLPRegressionHead(self.spec.feat_dim, out_dim=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, cooldown=3, min_lr=1e-6
        )
        # ReduceLROnPlateau is a Lightning monitored scheduler; tell
        # ``_scheduler_config`` what to watch.
        self.scheduler.interval = "epoch"
        self.scheduler.monitor = "val_loss"

        # Per-MLTK paradigm: model owns normalization. Datasets feed
        # un-normalized [0,1] tensors; AbstractModel auto-applies this
        # right before forward().
        self.normalize = transforms.Normalize(
            mean=self.spec.normalize_mean, std=self.spec.normalize_std,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.forward_backbone(image)
        return self.head(feats).squeeze(-1)

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Run the head only on already-extracted DINO features.

        Used by the cached-features training pipeline and by anything that
        already has a feature tensor in hand.
        """
        return self.head(features).squeeze(-1)

    def forward_backbone(self, image: torch.Tensor) -> torch.Tensor:
        """Run the frozen DINO backbone; returns the CLS-token features.
        Always runs under ``no_grad`` — backbone is frozen."""
        with torch.no_grad():
            return self.spec.forward(image)

    def get_focus_map(self, image: torch.Tensor) -> torch.Tensor:
        """CLS-to-patch from the last self-attention layer. Accepts an
        un-normalized [0,1] tensor and normalizes internally. Leaves the
        model in eval mode."""
        if self.spec.forward_with_attn is None:
            raise RuntimeError(
                f"Backbone {self.dino_model_name!r} does not expose attention "
                "maps (forward_with_attn is None)."
            )

        self.eval()
        with torch.no_grad():
            image = image.to(device=self.device, dtype=self.dtype)
            normed = self.normalize(image) if self.normalize is not None else image
            _, attn = self.spec.forward_with_attn(normed)

            prefix = self.spec.num_prefix_tokens or 1
            cls_to_patches = attn[:, :, 0, prefix:].mean(dim=1)

            patch_size = self.spec.patch_size or 14
            h = image.shape[2] // patch_size
            w = image.shape[3] // patch_size
            focus = cls_to_patches.reshape(-1, h, w)

            mn = focus.amin(dim=(1, 2), keepdim=True)
            mx = focus.amax(dim=(1, 2), keepdim=True)
            focus = (focus - mn) / (mx - mn + 1e-6)

            focus = torch.nn.functional.interpolate(
                focus.unsqueeze(1),
                size=(image.shape[2], image.shape[3]),
                mode='bicubic',
                align_corners=False,
            ).squeeze(1)
            return focus.clamp_(0.0, 1.0)


class DinoFeatureRegressionModel(AbstractModel):
    """Head-only Lightning module trained on **precomputed** DINO features.

    Mirror of ``DinoVisionAnalysisModel`` but with no backbone — inputs are
    feature vectors. Used during the cached-features training loop
    (``train_model.py``); inference still goes through the full
    image→backbone→head model.
    """

    def __init__(
        self,
        feat_dim: int,
        lr: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.head = MLPRegressionHead(feat_dim, out_dim=1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=5, cooldown=3, min_lr=1e-6
        )
        self.scheduler.interval = "epoch"
        self.scheduler.monitor = "val_loss"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features).squeeze(-1)
