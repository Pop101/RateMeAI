"""Train a regressor on top of a frozen DINOv3 backbone (Lightning).

Pipeline:
  1. ImageRatingDataset reads images and applies augmentation.
  2. ReplicatedDataset replays the source ``N`` times so each replica idx
     draws an independent random augment.
  3. MapDataset runs the inference model's ``forward_backbone`` to turn
     images into features.
  4. DiskCachedDataset writes those features to disk (mmap, chunked) so
     the backbone never runs at training time after the first epoch.
  5. ``DinoFeatureRegressionModel`` is a Lightning module trained on the
     cached features; ``Trainer.fit`` drives the loop.

The script runs in two phases by default:
  - **Phase 1 (smoke test, 10 batches)**: train for a handful of optimizer
    steps, save a checkpoint, reload it, and verify weights round-trip
    bit-for-bit. Confirms the cache + Lightning + checkpoint plumbing.
  - **Phase 2 (full training, until convergence)**: ``EarlyStopping`` on
    ``val_loss`` with patience 20.

DINOv3 weights are gated on HuggingFace. Before running:
  1. Accept the license at
     https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
  2. ``huggingface-cli login`` (or set ``HF_TOKEN`` in your environment).

Saved checkpoints contain only the head; ``evaluate_image.py`` keeps
running the full image -> backbone -> head pipeline at inference time
through ``DinoVisionAnalysisModel`` with ``strict=False`` head loading.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import modules  # noqa: F401  (side-effect: bootstrap MLTK on sys.path)
from modules.image_dataset import ImageRatingDataset
from modules.dino_model import DinoVisionAnalysisModel, DinoFeatureRegressionModel
from mltk import (
    DiskCachedDataset,
    MapDataset,
    ReplicatedDataset,
    auto_device,
    device_label,
)

device = auto_device()

IMAGE_SIZE = (224, 224)
N_AUGMENTED_REPLICAS = 4


def _backbone_slug(model) -> str:
    """Filesystem-safe identifier for the backbone, used to derive cache
    dir names so swapping backbones doesn't clobber prior caches."""
    name = (
        getattr(model, "dino_model_name", None)
        or getattr(model, "model_type", None)
        or getattr(model, "clip_model_name", None)
        or type(model).__name__
    )
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name))


def _build_cached_features(image_ds, model, *, replicas: int, cache_dir: str):
    """ReplicatedDataset → MapDataset(backbone) → DiskCachedDataset."""
    replicated = ReplicatedDataset(image_ds, n=replicas)

    def _to_features(img, lbl):
        # Backbone expects a batch dim; drop it back off so the cache
        # stores per-item feature vectors, not per-item batches.
        normed = model.normalize(img.to(device=device, dtype=model.dtype))
        feats = model.forward_backbone(normed.unsqueeze(0))
        return feats.squeeze(0).detach().cpu(), lbl

    feats = MapDataset(replicated, fn=_to_features)
    return DiskCachedDataset(feats, cache_dir=cache_dir, chunk_size=512)


def prepare_data(model, file_path: str, batch_size: int):
    df = pl.read_parquet(file_path)
    np.random.seed(42)
    df = df.with_columns(pl.lit(np.random.rand(df.shape[0])).alias("random"))
    train_df = df.filter(pl.col("random") <= 0.85).drop("random")
    test_df = df.filter(pl.col("random") > 0.85).drop("random")

    train_image_ds = ImageRatingDataset(train_df, size=IMAGE_SIZE, train=True)
    test_image_ds = ImageRatingDataset(test_df, size=IMAGE_SIZE, train=False)

    slug = _backbone_slug(model)
    h, w = IMAGE_SIZE
    train_cache_dir = os.path.join("cache", f"{slug}_{h}x{w}_train_x{N_AUGMENTED_REPLICAS}")
    test_cache_dir = os.path.join("cache", f"{slug}_{h}x{w}_test")

    train_cached = _build_cached_features(
        train_image_ds, model, replicas=N_AUGMENTED_REPLICAS, cache_dir=train_cache_dir
    )
    test_cached = _build_cached_features(
        test_image_ds, model, replicas=1, cache_dir=test_cache_dir
    )

    print(f"Precomputing train features -> {train_cache_dir}")
    train_cached.precompute(verbose=True)
    print(f"Precomputing test features -> {test_cache_dir}")
    test_cached.precompute(verbose=True)

    train_loader = DataLoader(train_cached, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_cached, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def _quiet_trainer(*, max_epochs: int = -1, max_steps: int = -1, callbacks=None) -> L.Trainer:
    accelerator = "gpu" if device.type in {"cuda", "mps"} else "cpu"
    return L.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator=accelerator,
        devices=1,
        callbacks=callbacks or [],
        default_root_dir="models",
        log_every_n_steps=10,
        enable_progress_bar=True,
    )


def smoke_test_save_load(
    head_model: DinoFeatureRegressionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    smoke_path: Path,
) -> None:
    """Train for a handful of steps; save and reload; verify weights match."""
    print("\n==== Phase 1: smoke test (10 optimizer steps) ====")
    trainer = _quiet_trainer(max_steps=10)
    trainer.fit(head_model, train_loader, val_loader)
    trainer.save_checkpoint(str(smoke_path))
    print(f"Saved smoke-test checkpoint -> {smoke_path}")

    reloaded = DinoFeatureRegressionModel.load_from_checkpoint(
        str(smoke_path), map_location="cpu",
    )
    print("Reloaded checkpoint OK; verifying head weights match...")
    for name, p_orig in head_model.head.named_parameters():
        p_loaded = dict(reloaded.head.named_parameters())[name]
        torch.testing.assert_close(
            p_orig.detach().cpu(), p_loaded.detach().cpu(),
            msg=f"head parameter {name!r} did not round-trip",
        )
    print("Save/load round-trip verified.\n")


def train_to_convergence(
    head_model: DinoFeatureRegressionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    out_path: Path,
    max_epochs: int,
    patience: int,
) -> None:
    print(f"==== Phase 2: training to convergence (max {max_epochs} epochs) ====")
    callbacks = [
        ModelCheckpoint(
            dirpath="models",
            filename="image_rating_model_{epoch:03d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=patience, mode="min"),
    ]
    trainer = _quiet_trainer(max_epochs=max_epochs, callbacks=callbacks)
    trainer.fit(head_model, train_loader, val_loader)
    trainer.save_checkpoint(str(out_path))
    print(f"Saved final checkpoint -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="reddit_posts_rated.parquet")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip the 10-step smoke test (assumes plumbing is healthy).")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)

    print(f"Device: {device_label()} ({device})")
    print("Loading DINOv3 backbone (this requires HF auth + license acceptance)...")
    backbone_holder = DinoVisionAnalysisModel(lr=args.lr)
    print(f"Backbone OK: feat_dim={backbone_holder.spec.feat_dim}, "
          f"image_size={backbone_holder.spec.image_size}")

    train_loader, test_loader = prepare_data(
        backbone_holder, file_path=args.data, batch_size=args.batch_size,
    )
    print(f"Training: {len(train_loader.dataset)} cached items "
          f"({len(train_loader.dataset) // N_AUGMENTED_REPLICAS} unique × "
          f"{N_AUGMENTED_REPLICAS} augments)")

    head_model = DinoFeatureRegressionModel(
        feat_dim=backbone_holder.spec.feat_dim, lr=args.lr,
    )

    if not args.skip_smoke:
        smoke_test_save_load(
            head_model, train_loader, test_loader,
            smoke_path=Path("models/image_rating_model_smoke.ckpt"),
        )
        # Re-instantiate to start convergence training from a fresh state —
        # the smoke test deliberately took a few steps with tiny effective
        # data, and we want convergence to start from initialization.
        head_model = DinoFeatureRegressionModel(
            feat_dim=backbone_holder.spec.feat_dim, lr=args.lr,
        )

    try:
        train_to_convergence(
            head_model, train_loader, test_loader,
            out_path=Path("models/image_rating_model_final.ckpt"),
            max_epochs=args.max_epochs, patience=args.patience,
        )
    except KeyboardInterrupt:
        print("Training interrupted.")


if __name__ == "__main__":
    main()
