import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

from modules.complex_head import ComplexHead, LayerNorm2d
from modules.torchgpu import device

class ConvnextModel:
    def __init__(self, lr=0.001):        
        # Initialize base model
        weights = ConvNeXt_Base_Weights.DEFAULT
        self.model = convnext_base(weights=weights)
        
        # Replace ConvNeXt's classifier with our complex head
        if hasattr(self.model, 'classifier') and isinstance(self.model.classifier[-1], nn.Linear):
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((2, 2)),
                LayerNorm2d(num_ftrs),
                nn.Flatten(1),
                ComplexHead(num_ftrs * 2 * 2, out_features=1),
                nn.Linear(1, 1),  # Scale layer
            )
            
        else:
            raise ValueError("Unsupported model architecture")
        
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
        labels = labels.to(device)  # Assuming labels are one-hot encoded tensors
        
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
                if transforms:
                    inputs = transforms(inputs)
                
                inputs = inputs.to(device)
                labels = labels.to(device)
            
                # Get output floats
                outputs = self.model(inputs)
                outputs = outputs.view(labels.shape)
                loss = self.criterion(outputs, labels)
                
                # Calculate accuracy for multi-label classification
                mae = torch.abs(outputs - labels).mean()
                
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_mae += mae.item() * batch_size
                num_samples += batch_size
        
        return total_loss / num_samples, total_mae / num_samples
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Return predicted probabilities for a single image"""
        self.model.eval()
        with torch.no_grad():
            return self.model(image)
        
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
    def load(filepath, lr=0.001):
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        
        model = ConvnextModel(lr=lr) # lr will be overwritten by the loaded value
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model