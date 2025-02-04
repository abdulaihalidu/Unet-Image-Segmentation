import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.model import UNet
from .train_utils import CombinedLoss


class Trainer:
    def __init__(self, config, train_loader, val_loader, save_dir):
        """
        Trainer class to handle training and validation of the UNet model.

        Args:
            config (dict): Configuration parameters.
            train_loader (DataLoader): Training dataset loader.
            val_loader (DataLoader): Validation dataset loader.
            save_dir (str or Path): Directory to save model checkpoints and plots.
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)  # Create save directory if it doesn't exist

        # Initialize model, loss function, optimizer, and scheduler
        self.model = self.initialize_model()
        self.criterion = CombinedLoss()
        self.optimizer, self.scheduler = self.get_optimizer()

        # Lists to store metrics over epochs
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.train_dices = []
        self.val_dices = []

    def initialize_model(self):
        """Initialize the UNet model with configured parameters."""
        model = UNet(
            encoder_channels=self.config.get("encoder_channels", (1, 64, 128, 256, 512, 1024)),
            decoder_channels=self.config.get("decoder_channels", (1024, 512, 256, 128, 64)),
            retain_dim=self.config.get("retain_dim", False),
            out_sz=self.config.get("out_size", (512, 512)),
            dropout_rate=self.config.get("dropout_rate", 0.5)
        )
        model.apply(self.init_weights)  # Apply weight initialization
        return model.to(self.device)

    @staticmethod
    def init_weights(m):
        """Apply Kaiming initialization to convolutional layers."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_optimizer(self):
        """Initialize Adam optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            betas=(self.config.get('beta1', 0.9), self.config.get('beta2', 0.999)),
            eps=self.config.get('eps', 1e-8),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        return optimizer, scheduler

    def train_epoch(self, epoch):
        """Perform one epoch of training."""
        self.model.train()
        epoch_loss = 0.0
        metrics = {'accuracy': 0.0, 'iou': 0.0, 'dice': 0.0}
        n_samples = 0

        for images, masks in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(images)  # Forward pass
            loss = self.criterion(outputs, masks)  # Compute loss
            loss.backward()  # Backpropagation
            self.optimizer.step()  # Update weights

            # Compute batch metrics
            batch_metrics = self.calculate_metrics(outputs, masks)
            batch_size = images.size(0)
            n_samples += batch_size
            epoch_loss += loss.item() * batch_size

            for key in metrics:
                metrics[key] += batch_metrics[key] * batch_size

        for key in metrics:
            metrics[key] /= n_samples
        metrics['loss'] = epoch_loss / n_samples
        return metrics

    def validate_epoch(self, epoch):
        """Perform one epoch of validation."""
        self.model.eval()
        epoch_loss = 0.0
        metrics = {'accuracy': 0.0, 'iou': 0.0, 'dice': 0.0}
        n_samples = 0

        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, masks)  # Compute loss
                batch_metrics = self.calculate_metrics(outputs, masks)  # Compute metrics
                batch_size = images.size(0)
                n_samples += batch_size
                epoch_loss += loss.item() * batch_size

                for key in metrics:
                    metrics[key] += batch_metrics[key] * batch_size

        for key in metrics:
            metrics[key] /= n_samples
        metrics['loss'] = epoch_loss / n_samples
        return metrics

    @staticmethod
    def calculate_metrics(outputs, targets, threshold=0.5):
        """Calculate segmentation accuracy, IoU, and Dice coefficient."""
        outputs = (torch.sigmoid(outputs) > threshold).float()  # Convert logits to binary mask

        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum() - intersection
        accuracy = (outputs == targets).float().mean().item() * 100
        iou = (intersection + 1e-7) / (union + 1e-7)
        dice = (2 * intersection + 1e-7) / (outputs.sum() + targets.sum() + 1e-7)

        return {'accuracy': accuracy, 'iou': iou.item(), 'dice': dice.item()}

    def save_metrics(self):
        """Plot and save training and validation loss, accuracy, and Dice scores."""
        plt.figure(figsize=(18, 6))

        # Loss curve
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Accuracy curve
        plt.subplot(1, 3, 2)
        plt.plot(self.train_accuracies, label="Train Accuracy")
        plt.plot(self.val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve")
        plt.legend()

        # Dice coefficient curve
        plt.subplot(1, 3, 3)
        plt.plot(self.train_dices, label="Train Dice")
        plt.plot(self.val_dices, label="Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice Coefficient")
        plt.title("Dice Curve")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / "metrics.png")
        plt.close()

    def train(self):
        """Main training loop with validation and model saving."""
        best_val_loss = float('inf')

        for epoch in range(1, self.config.get("end_epoch", 10) + 1):
            # Training and validation steps
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            # Update learning rate scheduler based on validation loss
            self.scheduler.step(val_metrics['loss'])

            # Store metrics
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            self.train_accuracies.append(train_metrics['accuracy'])
            self.val_accuracies.append(val_metrics['accuracy'])
            self.train_dices.append(train_metrics['dice'])
            self.val_dices.append(val_metrics['dice'])

            # Print epoch summary
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.2f}, "
                  f"Train Dice={train_metrics['dice']:.4f} | "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.2f}, "
                  f"Val Dice={val_metrics['dice']:.4f}")

            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                torch.save(self.model.state_dict(), self.save_dir / 'best_model.pth')

        # Save final training plots
        self.save_metrics()
        print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
