from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch

from PIL import Image

from typing import Tuple
import random

MEAN = [0.35657480359077454, 0.3154948949813843,  0.2955925762653351]
STD  = [0.32552820444107056, 0.30099010467529297, 0.2864872217178345]

def pad_to_square(img):
    width, height = img.size
    if width == height:
        return img
    
    size = max(width, height)
    result = Image.new("RGB", (size, size), (0, 0, 0))  # Black padding
    x_offset = (size - width) // 2
    y_offset = (size - height) // 2
    result.paste(img, (x_offset, y_offset))
    return result
        
class ImageRatingDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.to_dicts()
        self.transform = transform
    
    def get_transforms(self, train=True) -> transforms.Compose:
        """Create image transforms with learned mean and std"""        
        # Create transforms pipeline
        if train:
            return transforms.Compose([
                transforms.Lambda(pad_to_square),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        else:
            return transforms.Compose([
                transforms.Lambda(pad_to_square),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item["local_thumbnail_path"]
        rating = item["mean_rating"]
        
        # Load image. Note it is the user's job to apply transforms if desired
        image = Image.open(img_path).convert("RGB")
        return image, torch.tensor(rating, dtype=torch.float32)
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable image sizes"""
        images, ratings = zip(*batch)
        
        # Apply transforms if provided
        if self.transform:
            images = [self.transform(img) for img in images]
            images = torch.stack(images)
        else:
            # If no transforms provided, return list of PIL images
            images = [img.convert("RGB") for img in images]
        
        ratings = torch.stack(ratings)
        return images, ratings

# If we ever need to recalculate stats:
def calculate_stats(data:ImageRatingDataset, sample_size: int) -> Tuple[list, list]:
    """Calculate mean and std from a random sample of images"""
    # Take a sample of images
    sample_indices = random.sample(range(len(data.data)), min(sample_size, len(data.data)))
    
    # Convert to tensor (N, C, H, W)
    sample_tensors = []
    to_tensor = transforms.ToTensor()
    
    for idx in sample_indices:
        img_path = data.data[idx]["local_thumbnail_path"]
        img = Image.open(img_path).convert("RGB")
        img = pad_to_square(img)
        img = img.resize((224, 224))
        img_tensor = to_tensor(img)
        sample_tensors.append(img_tensor)
    
    # Stack tensors and calculate stats
    if sample_tensors:
        sample_batch = torch.stack(sample_tensors)
        mean = sample_batch.mean(dim=[0, 2, 3]).tolist()
        std = sample_batch.std(dim=[0, 2, 3]).tolist()
        return mean, std
    raise ValueError("No images found in the sample.")
    