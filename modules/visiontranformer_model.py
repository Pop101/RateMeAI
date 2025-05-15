import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from modules.complex_head import ComplexHead
from modules.torchgpu import device

class VisionTransformerModel:
    def __init__(self, input_size=224, lr=0.001):   
        # Initialize Vision Transformer     
        weights = ViT_B_16_Weights.DEFAULT
        self.model = vit_b_16(weights=weights)
        
        # Replace embedding layer with larger size
        if input_size != 224:
             self._modify_model_for_input_size(input_size)
        
        # Replace ViT's classifier head with our complex head
        hidden_dim = self.model.hidden_dim
        self.model.heads = ComplexHead(hidden_dim, out_features=1)
        
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
        
        model = VisionTransformerModel(lr=lr) # lr will be overwritten by the loaded value
        
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return model
    
    def _modify_model_for_input_size(self, input_size):
        """Modify the ViT model to accept a different input size"""
        # Update the image_size attribute to avoid assertion errors
        self.model.image_size = input_size
        
        # Save original positional embedding
        orig_pos_embed = self.model.encoder.pos_embedding
        
        # Original encoding size and patch size
        patch_size = self.model.patch_size
        orig_size = 224 // patch_size 
        new_size = input_size // patch_size
        
        # Extract class token and position embeddings
        cls_pos_embed = orig_pos_embed[:, 0:1]
        pos_embed = orig_pos_embed[:, 1:]
        
        # Reshape position embeddings to grid
        dim = pos_embed.shape[-1]
        pos_embed = pos_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        
        # Interpolate position embeddings to new size
        pos_embed = F.interpolate(
            pos_embed, 
            size=(new_size, new_size), 
            mode='bicubic', 
            align_corners=False
        )
        
        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        
        # Concat with class token and update model
        new_pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
        self.model.encoder.pos_embedding = nn.Parameter(new_pos_embed)
        
        # Update the expected sequence length
        self.model.seq_length = new_pos_embed.shape[1]