"""CLIP-ViT-as-frozen-backbone + ``MLPRegressionHead`` + attention map (Lightning).

For pure regression, prefer ``DinoVisionAnalysisModel`` — DINOv2 has
better spatial features. This subclass is kept around for the CLIP-flavored
attention map and as a comparison backbone. The backbone is loaded directly
via ``transformers.CLIPVisionModel`` because we want raw access to
``output_attentions=True`` on every forward.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from transformers import CLIPVisionModel

from modules.heads import MLPRegressionHead
from mltk import AbstractModel, auto_device

device = auto_device()

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CLIPVisionAnalysisModel(AbstractModel):
    def __init__(
        self,
        lr: float = 0.001,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.save_hyperparameters({"lr": lr, "clip_model_name": clip_model_name})
        self.clip_model_name = clip_model_name

        backbone = CLIPVisionModel.from_pretrained(clip_model_name)
        feat_dim = backbone.config.hidden_size
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
        self._backbone = backbone.to(device=device, dtype=dtype)

        self.head = MLPRegressionHead(feat_dim, out_dim=1)
        self.normalize = transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD)
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
            return self._backbone(pixel_values=image).pooler_output

    def get_focus_map(self, image: torch.Tensor) -> torch.Tensor:
        """Last-layer attention map. Leaves the model in eval mode."""
        self.eval()
        with torch.no_grad():
            image = image.to(device=self.device, dtype=self.dtype)
            normed = self.normalize(image) if self.normalize is not None else image
            outputs = self._backbone(pixel_values=normed, output_attentions=True)

            attn_weights = outputs.attentions[-1].mean(dim=1)
            patch_attn = attn_weights[:, 1:, 1:].mean(dim=1)

            patch_size = self._backbone.config.patch_size
            h = w = (image.shape[2] + patch_size - 1) // patch_size
            focus_mask = patch_attn.reshape(-1, h, w)

            mn = focus_mask.amin(dim=(1, 2), keepdim=True)
            mx = focus_mask.amax(dim=(1, 2), keepdim=True)
            focus_mask = (focus_mask - mn) / (mx - mn + 1e-6)

            focus_mask = torch.nn.functional.interpolate(
                focus_mask.unsqueeze(1),
                size=(image.shape[2], image.shape[3]),
                mode='bicubic',
                align_corners=False,
            ).squeeze(1)
            return focus_mask.clamp_(0.0, 1.0)
