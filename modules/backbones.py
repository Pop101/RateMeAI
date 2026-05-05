"""Shared vision-backbone loading for precompute + inference.

`BackboneSpec` is the single surface both `dev/data/precompute_features.py`
and `run.py` (the inference server) talk to, so we don't end up with two
subtly-different copies of "how to turn an image into a feature vector".

Supported families, detected from the HF model config:

    - SigLIP / SigLIP2 fixed-resolution          (SiglipVisionModel, Siglip2VisionModel)
    - SigLIP2 NaFlex (variable resolution,
      patch budget set by ``max_num_patches``)   (detected by model-id substring)
    - OpenAI CLIP / OpenCLIP                     (CLIPVisionModel)
    - DINOv2 / DINOv2-with-registers / DINOv3    (AutoModel + CLS-token pool)
    - Anything else (generic AutoModel
      + mean-pool over last_hidden_state)        (fallback; prints a warning)

`BackboneSpec.forward` has TWO signatures depending on `is_naflex`:

    - fixed-res:  forward(pixel_values: Tensor[B, 3, H, W]) -> Tensor[B, feat_dim]
    - NaFlex:     forward(batch: dict) -> Tensor[B, feat_dim]
                  batch keys: pixel_values, pixel_attention_mask, spatial_shapes

Callers branch on `spec.is_naflex` to build the right kind of input batch.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


@dataclass
class BackboneSpec:
    """Everything the caller needs to run a frozen vision backbone.

    Attention maps
    --------------
    For transformer backbones that expose self-attention (CLIP, DINOv2,
    DINOv2-with-registers, DINOv3), `forward_with_attn` returns the pooled
    features plus the last-layer self-attention tensor of shape
    ``[B, num_heads, seq_len, seq_len]``. The caller can then build a focus
    map however it wants — e.g. ``attn[:, :, 0, num_prefix_tokens:]`` for
    CLS-to-patch attention, reshaped to ``(H/patch, W/patch)``.

    `num_prefix_tokens` covers the leading non-patch tokens (CLS + register
    tokens), so the patch grid always starts at ``seq[num_prefix_tokens:]``.
    `patch_size` is the side length of one patch in pixels; the patch grid
    side is ``image_size // patch_size``.

    Backbones that don't expose attention (NaFlex, SigLIP pooling, generic
    fallback) leave `forward_with_attn=None` and `patch_size=0`.
    """

    hf_model_id: str
    slug: str
    image_size: int
    feat_dim: int
    normalize_mean: Tuple[float, float, float]
    normalize_std: Tuple[float, float, float]
    dtype: torch.dtype
    forward: Callable  # (tensor OR dict) -> [B, feat_dim]
    is_naflex: bool = False
    processor: object = None   # HF processor; NaFlex needs the real thing
    max_num_patches: int = 0   # only meaningful when is_naflex
    patch_size: int = 0        # side length of one patch in pixels (0 if N/A)
    num_prefix_tokens: int = 0 # leading non-patch tokens (CLS + registers)
    forward_with_attn: Optional[Callable] = None  # (tensor) -> (feats, last_layer_attn)

    @staticmethod
    def slug_for(hf_model_id: str) -> str:
        return hf_model_id.replace("/", "__")


def load_backbone(
    hf_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    max_num_patches: int = 1024,
) -> BackboneSpec:
    """Build a frozen BackboneSpec. The returned model lives on `device`
    in `dtype`, has `requires_grad=False` on every parameter, and is in
    eval mode. Downloading happens the first time a given model id is
    requested.
    """
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

    try:
        config = AutoConfig.from_pretrained(hf_model_id)
    except OSError as e:
        if "gated repo" in str(e).lower() or "401" in str(e):
            raise RuntimeError(
                f"Model {hf_model_id!r} is gated on HuggingFace. To use it:\n"
                f"  1. Accept the license at https://huggingface.co/{hf_model_id}\n"
                f"  2. Run 'huggingface-cli login' or set HF_TOKEN in your environment"
            ) from e
        raise
    vision_config = getattr(config, "vision_config", config)

    image_size = int(getattr(vision_config, "image_size", 224))
    feat_dim = int(getattr(vision_config, "hidden_size", 1024))

    model_type = getattr(config, "model_type", "") or getattr(vision_config, "model_type", "")
    lower = model_type.lower()
    is_naflex = "naflex" in hf_model_id.lower()

    if is_naflex:
        proc_full = AutoProcessor.from_pretrained(hf_model_id)
        image_proc = getattr(proc_full, "image_processor", proc_full)
    else:
        proc_full = None
        image_proc = AutoImageProcessor.from_pretrained(hf_model_id)
    processor = image_proc
    norm_mean = tuple(processor.image_mean) if hasattr(processor, "image_mean") else (0.5, 0.5, 0.5)
    norm_std = tuple(processor.image_std) if hasattr(processor, "image_std") else (0.5, 0.5, 0.5)

    # ----- NaFlex fast-path -----
    if is_naflex:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(hf_model_id, torch_dtype=dtype)
        vision = getattr(model, "vision_model", model).eval().to(device)
        for p in vision.parameters():
            p.requires_grad = False

        def forward_naflex(batch: dict) -> torch.Tensor:
            outputs = vision(
                pixel_values=batch["pixel_values"],
                attention_mask=batch["pixel_attention_mask"],
                spatial_shapes=batch["spatial_shapes"],
            )
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            hs = outputs.last_hidden_state  # [B, N, D]
            mask = batch["pixel_attention_mask"].to(dtype=hs.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (hs * mask).sum(dim=1) / denom

        from PIL import Image as _PILImage
        dummy = proc_full(
            images=[_PILImage.new("RGB", (448, 336), (128, 128, 128))],
            max_num_patches=max_num_patches,
            return_tensors="pt",
        )
        dummy_gpu = {
            k: v.to(device=device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
            for k, v in dummy.items()
        }
        with torch.no_grad():
            probe_out = forward_naflex(dummy_gpu)
        feat_dim = int(probe_out.size(-1))

        return BackboneSpec(
            hf_model_id=hf_model_id,
            slug=BackboneSpec.slug_for(hf_model_id),
            image_size=-1,
            feat_dim=feat_dim,
            normalize_mean=tuple(float(v) for v in norm_mean),
            normalize_std=tuple(float(v) for v in norm_std),
            dtype=dtype,
            forward=forward_naflex,
            is_naflex=True,
            processor=proc_full,
            max_num_patches=max_num_patches,
        )

    # ----- fixed-resolution backbones -----
    patch_size = int(getattr(vision_config, "patch_size", 0) or 0)
    num_prefix_tokens = 0
    forward_with_attn: Optional[Callable] = None

    if "siglip" in lower:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(hf_model_id, torch_dtype=dtype)
        vision = getattr(model, "vision_model", model).eval().to(device)

        def forward(pixel_values: torch.Tensor) -> torch.Tensor:
            outputs = vision(pixel_values=pixel_values)
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state.mean(dim=1)

        backbone_module = vision
        # SigLIP uses a learned-attention pooling head; raw attention maps
        # are not the typical interpretability signal, so leave unset.
    elif "clip" in lower:
        from transformers import CLIPVisionModel

        vision = CLIPVisionModel.from_pretrained(hf_model_id, torch_dtype=dtype).eval().to(device)

        def forward(pixel_values: torch.Tensor) -> torch.Tensor:
            return vision(pixel_values=pixel_values).pooler_output

        def forward_with_attn(pixel_values: torch.Tensor):
            outputs = vision(pixel_values=pixel_values, output_attentions=True)
            return outputs.pooler_output, outputs.attentions[-1]

        backbone_module = vision
        num_prefix_tokens = 1  # CLS token only
    elif "dinov" in lower:
        # Matches dinov2, dinov2_with_registers, dinov3, etc.
        from transformers import AutoModel

        vision = AutoModel.from_pretrained(hf_model_id, torch_dtype=dtype).eval().to(device)

        def forward(pixel_values: torch.Tensor) -> torch.Tensor:
            # CLS token is index 0 even with register tokens — registers come
            # after the CLS token in the sequence.
            return vision(pixel_values=pixel_values).last_hidden_state[:, 0]

        def forward_with_attn(pixel_values: torch.Tensor):
            outputs = vision(pixel_values=pixel_values, output_attentions=True)
            return outputs.last_hidden_state[:, 0], outputs.attentions[-1]

        backbone_module = vision
        # DINOv2 has 1 CLS token; DINOv2-with-registers / DINOv3 add register
        # tokens AFTER the CLS but BEFORE the patches.
        num_prefix_tokens = 1 + int(getattr(vision_config, "num_register_tokens", 0) or 0)
    else:
        from transformers import AutoModel

        vision = AutoModel.from_pretrained(hf_model_id, torch_dtype=dtype).eval().to(device)
        print(
            f"[backbones] warn: unknown model_type={model_type!r}; "
            "using generic mean-pooled last_hidden_state.",
            file=sys.stderr,
        )

        def forward(pixel_values: torch.Tensor) -> torch.Tensor:
            return vision(pixel_values=pixel_values).last_hidden_state.mean(dim=1)

        backbone_module = vision

    for p in backbone_module.parameters():
        p.requires_grad = False

    # Empirical probe so `feat_dim` matches reality even if the config lies.
    with torch.no_grad():
        probe = torch.zeros(1, 3, image_size, image_size, device=device, dtype=dtype)
        probe_out = forward(probe)
        if probe_out.dim() != 2 or probe_out.size(0) != 1:
            raise RuntimeError(
                f"Backbone {hf_model_id} produced unexpected output shape "
                f"{tuple(probe_out.shape)}; expected [1, feat_dim]."
            )
        feat_dim = int(probe_out.size(1))

    return BackboneSpec(
        hf_model_id=hf_model_id,
        slug=BackboneSpec.slug_for(hf_model_id),
        image_size=image_size,
        feat_dim=feat_dim,
        normalize_mean=tuple(float(v) for v in norm_mean),
        normalize_std=tuple(float(v) for v in norm_std),
        dtype=dtype,
        forward=forward,
        is_naflex=False,
        processor=processor,
        patch_size=patch_size,
        num_prefix_tokens=num_prefix_tokens,
        forward_with_attn=forward_with_attn,
    )
