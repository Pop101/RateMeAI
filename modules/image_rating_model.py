from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim

from efficientnet_pytorch import EfficientNet

from modules.torchgpu import device

BASE_MODEL = 'efficientnet-b0'

class ImageRatingModel:
    def __init__(self, model_type=BASE_MODEL, lr=0.001):
        # Initialize EfficientNet
        self.model = EfficientNet.from_pretrained(model_type)
        
        # Modify last layer for regression (as we want a linear output, not a softmax)
        # Use a simple but deeper perceptron head
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for m in self.model._fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Move to device
        self.model = self.model.to(device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=10,          
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
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(transformed_images)
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
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath, model_type=BASE_MODEL, lr=0.001):
        # Create new model instance
        model = ImageRatingModel(model_type=model_type, lr=lr)
        
        # Load saved state
        checkpoint = torch.load(filepath, map_location=device)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model
