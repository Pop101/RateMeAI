"""Reddit-rating image dataset.

This module is now a thin subclass of `mltk.AbstractImageDataset`. The base
owns the image pipeline (PIL -> tensor -> pad -> resize -> augment) and the
augmentation presets; we only implement `load_raw` to read a row out of the
parquet-derived list of dicts.

Normalization is *not* here on purpose — the model owns it (per-backbone
mean/std). See the MLTK paradigm note in `mltk/data/__init__.py`.

`pad_to_square` and `DeviceLRUCache` used to live here; they moved into
mltk. Kept as re-exports so existing `from modules.image_dataset import ...`
imports still work.
"""
import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from mltk import (  # noqa: F401
    AbstractImageDataset,
    DeviceLRUCache,
    STANDARD_AUGMENT,
    pad_to_square,
)

# Per-dataset stats kept here for the `calculate_stats` helper / archive value.
# They are NOT applied automatically anymore — the active model's normalize
# (in DinoVisionAnalysisModel) uses ImageNet stats from the backbone spec.
MEAN = [0.4907244145870209, 0.4381392002105713, 0.407711386680603]
STD  = [0.2914842963218689, 0.2807084321975708, 0.2719435691833496]


class ImageRatingDataset(AbstractImageDataset):
    # Windows-reserved chars in paths (PIL can't open them; URL query suffixes
    # like "image.jpg?3" survived the thumbnail download for ~0.5% of rows).
    _INVALID_PATH_CHARS = set('<>:"|?*')

    def __init__(
        self,
        dataframe,
        size: Tuple[int, int] = (320, 320),
        train: bool = True,
        augment: Optional[Callable] = STANDARD_AUGMENT,
    ):
        super().__init__(size=size, train=train, augment=augment)

        rows = dataframe.to_dicts()
        kept = []
        skipped_invalid = 0
        skipped_missing = 0
        for r in rows:
            p = r.get("local_path", "")
            if any(c in p for c in self._INVALID_PATH_CHARS):
                skipped_invalid += 1
                continue
            if not os.path.exists(p):
                skipped_missing += 1
                continue
            kept.append(r)
        if skipped_invalid or skipped_missing:
            print(
                f"ImageRatingDataset: kept {len(kept)}/{len(rows)} rows "
                f"(skipped {skipped_invalid} with invalid path chars, "
                f"{skipped_missing} missing on disk)"
            )
        self.data = kept

    def __len__(self) -> int:
        return len(self.data)

    def load_raw(self, idx: int):
        item = self.data[idx]
        img_path = item["local_path"]
        rating = (
            item["mean_rating"]
            if np.isnan(item["weighted_rating"]) or np.isinf(item["weighted_rating"])
            else item["weighted_rating"]
        )
        image = Image.open(img_path).convert("RGB")
        return image, torch.tensor(rating, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Stats helper — kept for archival use. Recompute if the corpus changes
# materially. Not run by anything in the training path.
# ---------------------------------------------------------------------------
def calculate_stats(data: ImageRatingDataset, sample_size: int):
    """Calculate channel mean/std from a random sample of images."""
    sample_indices = random.sample(range(len(data.data)), min(sample_size, len(data.data)))
    to_tensor = transforms.ToTensor()
    sample_tensors = []
    for idx in sample_indices:
        img_path = data.data[idx]["local_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(data.size)
        img_tensor = to_tensor(img)
        img_tensor = pad_to_square(img_tensor)
        sample_tensors.append(img_tensor)
    if sample_tensors:
        sample_batch = torch.stack(sample_tensors)
        mean = sample_batch.mean(dim=[0, 2, 3]).tolist()
        std = sample_batch.std(dim=[0, 2, 3]).tolist()
        return mean, std
    raise ValueError("No images found in the sample.")


if __name__ == "__main__":
    import polars as pl
    df = pl.read_parquet("./reddit_posts_rated.parquet")
    ds = ImageRatingDataset(df)
    print("Calculating stats...")
    mean, std = calculate_stats(ds, 800)
    print("Mean:", mean)
    print("STD:", std)
