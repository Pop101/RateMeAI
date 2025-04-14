import polars as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader

from modules.image_dataset import ImageRatingDataset
from modules.image_rating_model import ImageRatingModel

import itertools
from tqdm import tqdm
import os

# Prepare data
def prepare_data(file_path="data.parquet", batch_size=32, sample_size=100):
    # Load data
    df = pl.read_parquet(file_path)
    
    # Add random column for splitting
    df = df.with_columns(pl.lit(np.random.rand(df.shape[0])).alias("random"))
    
    # Split based on random value
    train_df = df.filter(pl.col("random") <= 0.8).drop("random")
    test_df = df.filter(pl.col("random") > 0.8).drop("random")
    
    # Create datasets
    # First create training dataset to learn stats
    train_dataset = ImageRatingDataset(train_df, transform=None)
    
    # Use the transforms from the training dataset for test dataset
    test_transforms = train_dataset.get_transforms(train=False)
    test_dataset = ImageRatingDataset(test_df, transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=test_dataset.collate_fn
    )
    
    return train_loader, test_loader, train_dataset

# Main function
def main():
    # Hyperparameters
    batch_size = 32
    sample_size = 100  # Number of images to sample for stats calculation
    
    # Prepare data
    train_loader, test_loader, train_dataset = prepare_data(
        file_path="reddit_posts_rated.parquet", 
        batch_size=batch_size,
        sample_size=sample_size
    )
    
    # Get transforms from dataset
    train_transforms = train_dataset.get_transforms(train=True)
    model = ImageRatingModel()
    os.makedirs("models", exist_ok=True)
    
    # Create infinite iterator over training data
    infinite_train_loader = itertools.cycle(train_loader)
    batch_count = 0
    
    print("Starting training...")
    print(f"Training on {len(train_loader.dataset)} images")
    try:
        pbar = tqdm(desc="Training", unit="batch")
        test_loss, test_mae = float("inf"), float("inf")
        while True:
            batch = next(infinite_train_loader)
            
            loss = model.train_batch(batch, transforms=train_transforms)
            batch_count += 1
            pbar.update(1)
            
            pbar.set_postfix({"Loss": f"{loss:.4f}", "Test Loss": f"{test_loss:.4f}", "Test MAE": f"{test_mae:.4f}"})

            if batch_count % 500 == 0:
                test_loss, test_mae = model.evaluate(test_loader)
                model.save(f"models/image_rating_model_batch_{batch_count}.pth")
                with open("models/losses.txt", "a") as f:
                    f.write(f"{batch_count},{loss},{test_loss},{test_mae}\n")
                
    except KeyboardInterrupt:
        pbar.close()
        print("Training interrupted. Saving final model...")
        model.save("models/image_rating_model_final.pth")
    print("Training complete.")
    
if __name__ == "__main__":
    main()