from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

from efficientnet_pytorch import EfficientNet

from modules.complex_head import ComplexHead
from modules.torchgpu import device

BASE_MODEL = 'efficientnet-b0'

class EfficientNetModel:
    """
    If you do not follow the table size, loss will be NaN
    B0: 224 x 224
    B1: 240 x 240
    B2: 260 x 260
    B3: 300 x 300
    B4: 380 x 380
    B5: 456 x 456
    B6: 528 x 528
    B7: 600 x 600
    """
    
    def __init__(self, model_type=BASE_MODEL, lr=0.001):
        # Initialize EfficientNet
        self.model = EfficientNet.from_pretrained(model_type)
        
        # Modify last layer for regression (as we want a linear output, not a softmax)
        # Use a simple but deeper perceptron head
        num_ftrs = self.model._fc.in_features
        self.model._fc = ComplexHead(num_ftrs, out_features=1)
        
        # Move to device
        self.model = self.model.to(device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-5,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,          
            cooldown=3,
            min_lr=1e-6
        )
    
    def train_batch(self, batch, transforms=None):
        self.model.train()
        
        # Unpack batch
        images, labels = batch
        
        # Apply transforms on-the-fly to each image if transforms provided
        if transforms:
            transformed_images = transforms(images)
        else:
            transformed_images = images            
        
        # Move to device
        transformed_images = transformed_images.to(device)
        labels = labels.to(device).view(-1, 1)
        
        if torch.isnan(transformed_images).any():
            print("NaN detected in input images!")
            return float('nan')
                    
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(transformed_images)
        outputs = outputs.squeeze()
        loss = self.criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_scheduler(self, val_loss):
        self.scheduler.step(val_loss)
    
    def evaluate(self, data_loader, transforms=None):
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)
                
                if transforms:
                    inputs = transforms(inputs)
                
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, labels)
                
                # Calculate MAE
                mae = torch.abs(outputs - labels).mean()
                
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                num_samples += batch_size
        
        return total_loss / num_samples, total_mae / num_samples
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return predicted rating for a single image"""
        self.model.eval()
        with torch.no_grad():
            image = image.to(device)
            output = self.model(image)
            return output
        
    def get_current_lr(self):
        """Return the current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def send_to_device(self, device):
        """Sends the current model to the specified device, mutating the model (not like .to)"""
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
    
    @staticmethod
    def load(filepath, model_type=BASE_MODEL, lr=0.001):
        # Create new model instance
        model = EfficientNetModel(model_type=model_type, lr=lr)
        
        # Load saved state
        checkpoint = torch.load(filepath, map_location=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model
