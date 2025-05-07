from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import numpy as np

from PIL import Image

from collections import OrderedDict

from typing import Tuple
import random

MEAN = [0.4937741160392761, 0.4392690062522888, 0.40500015020370483]
STD  = [0.29177841544151306, 0.27624377608299255, 0.26687321066856384]

def pad_to_square(image:torch.Tensor) -> torch.Tensor:
    """Pad an image to make it square."""
    c, h, w = image.shape
    
    # Find the maximum dimension
    max_dim = max(h, w)
    
    # Calculate padding for height and width
    pad_h = (max_dim - h) // 2
    pad_h_remainder = (max_dim - h) % 2  # Handle odd padding
    
    pad_w = (max_dim - w) // 2
    pad_w_remainder = (max_dim - w) % 2  # Handle odd padding
    
    # Apply padding [left, right, top, bottom]
    padding = (pad_w, pad_w + pad_w_remainder, pad_h, pad_h + pad_h_remainder)
    
    # Use torch functional padding with constant value (usually 0 or 1 depending on normalization)
    padded_image = torch.nn.functional.pad(image, padding, mode='constant', value=0)
    return padded_image
        
class ImageRatingDataset(Dataset):
    def __init__(self, dataframe, size:tuple=(320, 320)):
        self.data = dataframe.to_dicts()
        self.size = size
    
    @staticmethod
    def get_transforms(train=True) -> transforms.Compose:
        """Create image transforms with learned mean and std"""        
        # Create transforms pipeline
        if train:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        else:
            return transforms.Compose([
                transforms.Normalize(mean=MEAN, std=STD),
            ])
            
    def get_size_transpose(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(pad_to_square),
            transforms.Resize(self.size),
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["local_path"]
        rating = item["mean_rating"] if np.isnan(item["weighted_rating"]) or np.isinf(item["weighted_rating"]) else item["weighted_rating"]
        
        # Load image. Note it is the user's job to apply transforms if desired
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.get_size_transpose()(image)
        return image_tensor, torch.tensor(rating, dtype=torch.float32)



class LRUCacheDataset(Dataset):
    def __init__(self, dataset, cache_size=5000, device='cuda'):
        self.dataset = dataset
        self.cache_size = cache_size
        self.device = device
        self.cache = OrderedDict()  # LRU cache: idx -> (image, label)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            # Move item to end (most recently used)
            self.cache.move_to_end(idx)
            return self.cache[idx]
        
        # Get from original dataset
        image, label = self.dataset[idx]
        
        # Move to device
        image = image.to(self.device)
        label = label.to(self.device)
        
        # Add to cache
        self.cache[idx] = (image, label)
        
        # Remove oldest item if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        
        return image, label


# If we ever need to recalculate stats:
def calculate_stats(data:ImageRatingDataset, sample_size: int) -> Tuple[list, list]:
    """Calculate mean and std from a random sample of images"""
    # Take a sample of images
    sample_indices = random.sample(range(len(data.data)), min(sample_size, len(data.data)))
    
    # Convert to tensor (N, C, H, W)
    sample_tensors = []
    to_tensor = transforms.ToTensor()
    
    for idx in sample_indices:
        img_path = data.data[idx]["local_path"]
        img = Image.open(img_path).convert("RGB")
        img = img.resize(data.size)
        img_tensor = to_tensor(img)
        img_tensor = pad_to_square(img_tensor)
        sample_tensors.append(img_tensor)
    
    # Stack tensors and calculate stats
    if sample_tensors:
        sample_batch = torch.stack(sample_tensors)
        mean = sample_batch.mean(dim=[0, 2, 3]).tolist()
        std = sample_batch.std(dim=[0, 2, 3]).tolist()
        return mean, std
    raise ValueError("No images found in the sample.")

if __name__ == "__main__":
    # Recalculate stats
    import polars as pl
    df = pl.read_parquet("./reddit_posts_rated.parquet")
    ds = ImageRatingDataset(df)
    
    print("Calculating stats...")
    
    mean, std = calculate_stats(ds, 800)
    print("Mean:", mean)
    print("STD:", std)