"""Training utilities for protein structure prediction models."""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import EarlyStopping, save_checkpoint, load_checkpoint


class ProteinTrainer:
    """Trainer class for protein structure prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        config: Dict,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to use
            config: Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.001),
            restore_best_weights=True,
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Move model to device
        self.model.to(device)
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(outputs, batch.y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get("grad_clip", 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip"])
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100 * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self) -> Tuple[float, float]:
        """Validate for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(outputs, batch.y)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def test(self) -> Tuple[float, float, Dict[str, float]]:
        """Test the model.
        
        Returns:
            Tuple of (loss, accuracy, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(outputs, batch.y)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = correct / total
        
        # Calculate additional metrics
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        return avg_loss, accuracy, metrics
    
    def _calculate_metrics(self, predictions: List[int], targets: List[int]) -> Dict[str, float]:
        """Calculate additional metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
        
        # Convert to numpy arrays
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        
        # Calculate metrics
        precision = precision_score(targets, predictions, average="weighted")
        recall = recall_score(targets, predictions, average="weighted")
        f1 = f1_score(targets, predictions, average="weighted")
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        
        return metrics
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} parameters")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss,
                    {"accuracy": val_acc}, checkpoint_path
                )
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }
    
    def load_best_model(self, checkpoint_path: str) -> None:
        """Load the best model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        epoch, loss, metrics = load_checkpoint(checkpoint_path, self.model, self.optimizer)
        print(f"Loaded model from epoch {epoch} with validation loss {loss:.4f}")
        print(f"Validation metrics: {metrics}")


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer for the model.
    
    Args:
        model: Model to optimize
        optimizer_name: Name of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer
    """
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            **kwargs
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = "cosine",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        scheduler_name: Name of scheduler
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 0),
        )
    elif scheduler_name.lower() == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get("step_size", 30),
            gamma=kwargs.get("gamma", 0.1),
        )
    elif scheduler_name.lower() == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
        )
    else:
        return None
