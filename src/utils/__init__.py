"""Utility functions for protein structure prediction with GNNs."""

import random
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        torch.device: Available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        metrics: Evaluation metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "metrics": metrics,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Tuple[int, float, Dict[str, float]]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model
        optimizer: Optional optimizer
        
    Returns:
        Tuple of (epoch, loss, metrics)
    """
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"], checkpoint["loss"], checkpoint["metrics"]


def calculate_rmsd(
    pred_coords: torch.Tensor, 
    true_coords: torch.Tensor
) -> torch.Tensor:
    """Calculate Root Mean Square Deviation between predicted and true coordinates.
    
    Args:
        pred_coords: Predicted coordinates [N, 3]
        true_coords: True coordinates [N, 3]
        
    Returns:
        RMSD value
    """
    return torch.sqrt(torch.mean(torch.sum((pred_coords - true_coords) ** 2, dim=1)))


def calculate_mae(
    pred_values: torch.Tensor, 
    true_values: torch.Tensor
) -> torch.Tensor:
    """Calculate Mean Absolute Error.
    
    Args:
        pred_values: Predicted values
        true_values: True values
        
    Returns:
        MAE value
    """
    return torch.mean(torch.abs(pred_values - true_values))


def calculate_contact_map_accuracy(
    pred_contacts: torch.Tensor,
    true_contacts: torch.Tensor,
    threshold: float = 8.0,
) -> torch.Tensor:
    """Calculate contact map prediction accuracy.
    
    Args:
        pred_contacts: Predicted contact map
        true_contacts: True contact map
        threshold: Distance threshold for contacts
        
    Returns:
        Contact map accuracy
    """
    pred_binary = (pred_contacts < threshold).float()
    true_binary = (true_contacts < threshold).float()
    
    correct = (pred_binary == true_binary).float()
    return torch.mean(correct)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights
            
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module) -> None:
        """Save model weights."""
        self.best_weights = model.state_dict().copy()
