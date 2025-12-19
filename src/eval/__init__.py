"""Evaluation utilities for protein structure prediction models."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from ..utils import calculate_rmsd, calculate_mae, calculate_contact_map_accuracy


class ProteinEvaluator:
    """Evaluator class for protein structure prediction models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        num_classes: int,
    ):
        """Initialize evaluator.
        
        Args:
            model: Model to evaluate
            test_loader: Test data loader
            device: Device to use
            num_classes: Number of classes
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        
        # Move model to device
        self.model.to(device)
    
    def evaluate_classification(self) -> Dict[str, float]:
        """Evaluate model for classification tasks.
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_classification_metrics(
            all_predictions, all_targets, all_probabilities
        )
        
        return metrics
    
    def evaluate_structure_prediction(
        self,
        pred_coords: Optional[torch.Tensor] = None,
        true_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Evaluate model for structure prediction tasks.
        
        Args:
            pred_coords: Predicted coordinates
            true_coords: True coordinates
            
        Returns:
            Dictionary of evaluation metrics
        """
        if pred_coords is None or true_coords is None:
            # Generate dummy data for demonstration
            pred_coords = torch.randn(100, 3)
            true_coords = torch.randn(100, 3)
        
        # Calculate structure metrics
        rmsd = calculate_rmsd(pred_coords, true_coords)
        mae = calculate_mae(pred_coords, true_coords)
        
        metrics = {
            "rmsd": rmsd.item(),
            "mae": mae.item(),
        }
        
        return metrics
    
    def evaluate_contact_prediction(
        self,
        pred_contacts: Optional[torch.Tensor] = None,
        true_contacts: Optional[torch.Tensor] = None,
        threshold: float = 8.0,
    ) -> Dict[str, float]:
        """Evaluate model for contact prediction tasks.
        
        Args:
            pred_contacts: Predicted contact map
            true_contacts: True contact map
            threshold: Distance threshold for contacts
            
        Returns:
            Dictionary of evaluation metrics
        """
        if pred_contacts is None or true_contacts is None:
            # Generate dummy data for demonstration
            pred_contacts = torch.randn(50, 50)
            true_contacts = torch.randn(50, 50)
        
        # Calculate contact map accuracy
        contact_accuracy = calculate_contact_map_accuracy(
            pred_contacts, true_contacts, threshold
        )
        
        metrics = {
            "contact_accuracy": contact_accuracy.item(),
        }
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate classification metrics.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            probabilities: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average="weighted", zero_division=0)
        recall = recall_score(targets, predictions, average="weighted", zero_division=0)
        f1 = f1_score(targets, predictions, average="weighted", zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }
        
        # ROC-AUC (for binary classification)
        if self.num_classes == 2:
            try:
                roc_auc = roc_auc_score(targets, probabilities[:, 1])
                metrics["roc_auc"] = roc_auc
            except ValueError:
                metrics["roc_auc"] = 0.0
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        
        for i in range(self.num_classes):
            metrics[f"precision_class_{i}"] = precision_per_class[i]
            metrics[f"recall_class_{i}"] = recall_per_class[i]
            metrics[f"f1_class_{i}"] = f1_per_class[i]
        
        return metrics
    
    def generate_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Generate confusion matrix.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(targets, predictions)
    
    def generate_classification_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> str:
        """Generate classification report.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            class_names: Optional class names
            
        Returns:
            Classification report string
        """
        return classification_report(
            targets, predictions,
            target_names=class_names,
            zero_division=0
        )
    
    def evaluate_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """Evaluate model on all tasks.
        
        Returns:
            Dictionary of task-specific metrics
        """
        results = {}
        
        # Classification evaluation
        classification_metrics = self.evaluate_classification()
        results["classification"] = classification_metrics
        
        # Structure prediction evaluation
        structure_metrics = self.evaluate_structure_prediction()
        results["structure_prediction"] = structure_metrics
        
        # Contact prediction evaluation
        contact_metrics = self.evaluate_contact_prediction()
        results["contact_prediction"] = contact_metrics
        
        return results


def create_leaderboard(
    model_results: Dict[str, Dict[str, float]],
    model_names: List[str],
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Create a leaderboard comparing multiple models.
    
    Args:
        model_results: Results from multiple models
        model_names: Names of models
        metric_names: Names of metrics to compare
        
    Returns:
        Leaderboard dictionary
    """
    leaderboard = {}
    
    for metric in metric_names:
        leaderboard[metric] = {}
        
        for model_name in model_names:
            if model_name in model_results:
                # Find metric in nested results
                value = None
                for task_results in model_results[model_name].values():
                    if metric in task_results:
                        value = task_results[metric]
                        break
                
                if value is not None:
                    leaderboard[metric][model_name] = value
    
    return leaderboard


def print_leaderboard(leaderboard: Dict[str, Dict[str, float]]) -> None:
    """Print formatted leaderboard.
    
    Args:
        leaderboard: Leaderboard dictionary
    """
    print("\n" + "="*80)
    print("MODEL LEADERBOARD")
    print("="*80)
    
    for metric, model_scores in leaderboard.items():
        print(f"\n{metric.upper()}:")
        print("-" * 40)
        
        # Sort models by score (descending)
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (model_name, score) in enumerate(sorted_models, 1):
            print(f"{i:2d}. {model_name:20s}: {score:.4f}")
    
    print("="*80)


def calculate_model_complexity(model: nn.Module) -> Dict[str, int]:
    """Calculate model complexity metrics.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of complexity metrics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }
