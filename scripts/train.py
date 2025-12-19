#!/usr/bin/env python3
"""Main training script for protein structure prediction with GNNs."""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data import load_protein_dataset, ProteinDataProcessor
from src.models import ProteinGCN, ProteinGAT, ProteinEGNN, ProteinGIN
from src.train import ProteinTrainer, create_optimizer, create_scheduler
from src.eval import ProteinEvaluator, print_leaderboard
from src.utils import set_seed, get_device, count_parameters


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    """Create model based on configuration.
    
    Args:
        config: Model configuration
        input_dim: Input feature dimension
        output_dim: Output dimension
        
    Returns:
        Model instance
    """
    model_config = config["model"]
    model_type = model_config["type"]
    
    if model_type == "ProteinGCN":
        return ProteinGCN(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            output_dim=output_dim,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            use_batch_norm=model_config["use_batch_norm"],
        )
    elif model_type == "ProteinGAT":
        return ProteinGAT(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            output_dim=output_dim,
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            dropout=model_config["dropout"],
            use_batch_norm=model_config["use_batch_norm"],
        )
    elif model_type == "ProteinEGNN":
        return ProteinEGNN(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            output_dim=output_dim,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
        )
    elif model_type == "ProteinGIN":
        return ProteinGIN(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            output_dim=output_dim,
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            use_batch_norm=model_config["use_batch_norm"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train protein structure prediction model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, mps, cpu, auto)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(config["paths"]["logs_dir"], exist_ok=True)
    os.makedirs(config["paths"]["assets_dir"], exist_ok=True)
    os.makedirs(config["paths"]["results_dir"], exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    train_loader, val_loader, test_loader, num_features, num_classes = load_protein_dataset(
        root=config["data"]["root_dir"],
        name=config["data"]["dataset_name"],
        train_split=config["data"]["train_split"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        batch_size=config["train"]["batch_size"],
        shuffle=config["data"]["shuffle"],
    )
    
    print(f"Dataset loaded: {num_features} features, {num_classes} classes")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config, num_features, num_classes)
    print(f"Model created: {count_parameters(model)} parameters")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        optimizer_name=config["train"]["optimizer"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )
    
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=config["train"]["scheduler"],
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = ProteinTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config["train"],
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_best_model(args.resume)
    
    # Train model
    print("Starting training...")
    history = trainer.train(config["train"]["num_epochs"])
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = ProteinEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
    )
    
    test_loss, test_accuracy, test_metrics = evaluator.test()
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  Metrics: {test_metrics}")
    
    # Save final model
    checkpoint_path = os.path.join(config["paths"]["checkpoints_dir"], "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_metrics": test_metrics,
        "config": config,
    }, checkpoint_path)
    
    print(f"Model saved to: {checkpoint_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
