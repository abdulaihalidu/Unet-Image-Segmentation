import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm 

from models.model import UNet


# Loss function combining Binary Cross Entropy (BCE) and Dice Loss
class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_criterion = nn.BCEWithLogitsLoss()  # BCE Loss with logits

    def forward(self, inputs, targets):
        bce_loss = self.bce_criterion(inputs, targets)
        
        # Dice loss computation
        inputs_sigmoid = torch.sigmoid(inputs)  # Apply sigmoid to logits
        intersection = (inputs_sigmoid * targets).sum()
        dice_loss = 1 - (2. * intersection + 1e-7) / (
            inputs_sigmoid.sum() + targets.sum() + 1e-7
        )
        
        # Return weighted combination of BCE and Dice loss
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def compute_statistics(self, dataset_path='dataset/train'):
        """Compute mean and standard deviation of the training dataset"""
        sum_pixels = 0
        sum_squared_pixels = 0
        num_pixels = 0 

        imgs_dir = os.path.join(dataset_path, 'imgs')
        image_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        
        for img_file in image_files:
            img_path = os.path.join(dataset_path, 'imgs', img_file)
            image = np.array(Image.open(img_path).convert('L')).astype(np.float32) / 255.0
            sum_pixels += image.sum()
            sum_squared_pixels += (image ** 2).sum()
            num_pixels += image.size
            
        mean = sum_pixels / num_pixels
        std = np.sqrt(sum_squared_pixels / num_pixels - mean ** 2)
        return mean, std

    def calculate_metrics(self, logit, target, threshold=0.5):
        """Calculate segmentation metrics: Accuracy, IoU, and Dice coefficient"""
        preds = (logit > threshold).float()

        # Compute Accuracy
        acc = (preds == target).float().mean().item() * 100
        
        # Compute IoU (Intersection over Union)
        intersection = (preds * target).sum()
        union = preds.sum() + target.sum() - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        # Compute Dice coefficient
        dice = (2 * intersection + 1e-7) / (preds.sum() + target.sum() + 1e-7)
        
        return {
            'accuracy': acc,
            'iou': iou.item(),
            'dice': dice.item()
        }

    def step(self, data_loader, split, epoch, model, criterion, optimizer=None, 
             batch_size=1, max_grad_norm=1.0):
        """Single training/evaluation step for one epoch"""
        model.train() if split == 'train' else model.eval()
        
        metrics_dict = {
            'loss': 0.0,
            'accuracy': 0.0,
            'iou': 0.0,
            'dice': 0.0
        }
        n_samples = 0
        
        pbar = tqdm(data_loader, desc=f'Epoch {epoch} ({split})')
        
        for batch_idx, (image, label) in enumerate(pbar):
            try:
                image, label = image.to(self.device), label.to(self.device)
                
                with torch.set_grad_enabled(split == 'train'):
                    logit = model(image)  # Forward pass
                    loss = criterion(logit, label)
                    
                    if split == 'train':  # Backpropagation for training
                        optimizer.zero_grad()
                        loss.backward()
                        
                        if max_grad_norm > 0:  # Gradient clipping
                            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        
                        optimizer.step()
                
                # Compute batch metrics
                batch_metrics = self.calculate_metrics(logit, label)
                batch_size = image.size(0)
                
                # Accumulate loss and metrics
                metrics_dict['loss'] += loss.item() * batch_size
                metrics_dict['accuracy'] += batch_metrics['accuracy'] * batch_size
                metrics_dict['iou'] += batch_metrics['iou'] * batch_size
                metrics_dict['dice'] += batch_metrics['dice'] * batch_size
                n_samples += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': metrics_dict['loss'] / n_samples,
                    'acc': metrics_dict['accuracy'] / n_samples,
                    'dice': metrics_dict['dice'] / n_samples
                })
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                if "out of memory" in str(e) and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                continue
        
        # Normalize metrics by number of samples
        for key in metrics_dict:
            metrics_dict[key] /= n_samples
        
        return metrics_dict

    def initialize_model(self):
        """Initialize U-Net model with weight initialization"""
        model = UNet(
            retain_dim=True,
            out_sz=self.config.get('out_size', (512, 512))
        )
        
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        model.apply(init_weights)
        return model.to(self.device)

    def get_optimizer(self, model):
        """Create Adam optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
            eps=self.config.get('eps', 1e-8),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5
        )
        return optimizer, scheduler

    def save_metrics(self, save_dir):
        """Plot and save learning curves"""
        plt.figure(figsize=(15, 5))
        
        # Plot loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Learning Curves - Loss')
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Learning Curves - Accuracy')
        
        plt.savefig(save_dir / 'learning_curves.png')
        plt.close()

    def train(self, train_loader, val_loader):
        """Main training loop"""
        save_dir = Path('../training/checkpoints')
        save_dir.mkdir(exist_ok=True)
        
        model, criterion, optimizer, scheduler = self.initialize_model(), CombinedLoss(), *self.get_optimizer(self.initialize_model())
        best_val_loss = float('inf')
        
        for epoch in range(self.config['start_epoch'], self.config['end_epoch']):
            # Training step
            train_metrics = self.step(train_loader, 'train', epoch, model, criterion, optimizer)
            
            # Validation step
            with torch.no_grad():
                val_metrics = self.step(val_loader, 'val', epoch, model, criterion)
            
            scheduler.step(val_metrics['loss'])  # Adjust learning rate
            
            # Save model if validation loss improves
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(model.state_dict(), save_dir / f'model_best.pth')
        
        return model, best_val_loss
