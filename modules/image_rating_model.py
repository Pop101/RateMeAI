import torch
import torch.nn as nn
import torch.optim as optim

from efficientnet_pytorch import EfficientNet

from modules.torchgpu import device
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ImageRatingModel:
    def __init__(self, model_type='efficientnet-b3', lr=0.001):
        # Initialize EfficientNet
        self.model = EfficientNet.from_pretrained(model_type)
        
        # Modify last layer for regression (as we want a linear output, not a softmax)
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Linear(num_ftrs, 1)
        
        # Move to device
        self.model = self.model.to(device)
        
        # Initialize criterion and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',           # Reduce LR when the validation loss stops decreasing
            factor=0.1,           # Multiply the learning rate by this factor
            patience=5,           # Number of epochs with no improvement after which LR will be reduced
            verbose=True,         # Print message when LR is reduced
            min_lr=1e-6           # Lower bound on the learning rate
        )
    
    def train_batch(self, batch, transforms=None):
        self.model.train()
        
        # Unpack batch
        images, labels = batch
        images.to(device)
        labels.to(device)
        
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
    
    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Calculate MAE
                mae = torch.abs(outputs - labels).mean()
                
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                num_samples += batch_size
        
        return total_loss / num_samples, total_mae / num_samples
    
    def get_current_lr(self):
        """Return the current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # Save scheduler state
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath, model_type='efficientnet-b0', lr=0.001):
        # Create new model instance
        model = ImageRatingModel(model_type=model_type, lr=lr)
        
        # Load saved state
        checkpoint = torch.load(filepath)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint:
            model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model