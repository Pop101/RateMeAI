import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

from modules.image_dataset import ImageRatingDataset, LRUCacheDataset
from modules.image_rating_model import ImageRatingModel
from modules.torchgpu import device

import itertools
from tqdm import tqdm
import os
import shutil

# Prepare data
def prepare_data(file_path="data.parquet", batch_size=32):
    # Load data
    df = pl.read_parquet(file_path)
    
    # Add random column for splitting
    np.random.seed(42)  # For reproducibility
    df = df.with_columns(pl.lit(np.random.rand(df.shape[0])).alias("random"))
    
    # Split based on random value
    train_df = df.filter(pl.col("random") <= 0.85).drop("random")
    test_df = df.filter(pl.col("random") > 0.85).drop("random")
    
    # Create datasets
    # First create training dataset to learn stats
    train_dataset = ImageRatingDataset(train_df)
    train_dataset = LRUCacheDataset(train_dataset, cache_size=10000, device=device)
    
    test_dataset = ImageRatingDataset(test_df)
    test_dataset = LRUCacheDataset(test_dataset, cache_size=10000, device=device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0,
    )
    
    return train_loader, test_loader

# Main function
def main():    
    # Hyperparameters
    batch_size = 32
    
    # Prepare data
    train_loader, test_loader = prepare_data(
        file_path="reddit_posts_rated.parquet", 
        batch_size=batch_size,
    )
    
    # Get transforms from dataset
    train_transforms = ImageRatingDataset.get_transforms(train=True)
    test_transforms  = ImageRatingDataset.get_transforms(train=False)
    
    model = ImageRatingModel()
    shutil.rmtree("models", ignore_errors=True)
    os.makedirs("models", exist_ok=True)
    
    print("Starting training...")
    print(f"Training on {len(train_loader.dataset)} images")
    try:
        pbar = tqdm(desc="Training", unit="batch")
        test_loss, test_mae = float("inf"), float("inf")
        batch_count = 0
        
        while True:
            for batch in train_loader:
                loss = model.train_batch(batch, transforms=train_transforms)
                torch.cuda.synchronize()
                batch_count += 1
                pbar.update(1)
                
                pbar.set_postfix({"Loss": f"{loss:.4f}", "Test Loss": f"{test_loss:.4f}", "Test MAE": f"{test_mae:.4f}"})
                if batch_count % 100 == 0:
                    test_loss, test_mae = model.evaluate(tqdm(test_loader, desc="Testing", unit="batch"), transforms=test_transforms)
                    model.update_scheduler(test_loss)
                    with open("models/losses.txt", "a") as f:
                        f.write(f"{batch_count},{model.get_current_lr()},{loss},{test_loss},{test_mae}\n")
                    
                if batch_count % 500 == 0:
                    print('\n')
                    model.save(f"models/image_rating_model_batch_{batch_count}.pth")
                    
                if batch_count > 25000:
                    # Yeah the model has DEFINITELY converged by now
                    break
 
    except KeyboardInterrupt:
        print("Training interrupted. Saving final model...")
    finally:
        pbar.close()
        model.save("models/image_rating_model_final.pth")
    
    print("Training complete.")
    
if __name__ == "__main__":
    main()