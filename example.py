#!/usr/bin/env python3
"""Example script demonstrating the modernized protein structure prediction with GNNs."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.data import create_synthetic_protein_dataset, ProteinDataProcessor
from src.models import ProteinGCN, ProteinGAT, ProteinEGNN, ProteinGIN
from src.train import create_optimizer
from src.eval import ProteinEvaluator
from src.utils import set_seed, get_device, count_parameters


def main():
    """Main example function."""
    print("🧬 Protein Structure Prediction with Graph Neural Networks")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create synthetic protein dataset
    print("\n📊 Creating synthetic protein dataset...")
    proteins = create_synthetic_protein_dataset(
        num_proteins=50,
        min_residues=20,
        max_residues=50,
        num_classes=3,
        num_features=20
    )
    
    print(f"Created {len(proteins)} protein graphs")
    print(f"Example protein: {proteins[0].x.shape[0]} residues, {proteins[0].edge_index.shape[1]} edges")
    
    # Process data
    processor = ProteinDataProcessor(
        add_positional_encoding=True,
        normalize_features=True,
        add_edge_attributes=True
    )
    
    processed_proteins = [processor(protein) for protein in proteins]
    print(f"Processed proteins with {processed_proteins[0].x.shape[1]} features per residue")
    
    # Create data loaders
    train_size = int(0.7 * len(processed_proteins))
    val_size = int(0.15 * len(processed_proteins))
    
    train_proteins = processed_proteins[:train_size]
    val_proteins = processed_proteins[train_size:train_size + val_size]
    test_proteins = processed_proteins[train_size + val_size:]
    
    train_loader = DataLoader(train_proteins, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_proteins, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_proteins, batch_size=8, shuffle=False)
    
    print(f"Data splits: Train={len(train_proteins)}, Val={len(val_proteins)}, Test={len(test_proteins)}")
    
    # Test different models
    models = {
        "ProteinGCN": ProteinGCN(input_dim=processed_proteins[0].x.shape[1], hidden_dim=32, output_dim=3),
        "ProteinGAT": ProteinGAT(input_dim=processed_proteins[0].x.shape[1], hidden_dim=32, output_dim=3, num_heads=4),
        "ProteinGIN": ProteinGIN(input_dim=processed_proteins[0].x.shape[1], hidden_dim=32, output_dim=3),
    }
    
    print("\n🤖 Testing different GNN architectures...")
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print(f"  Parameters: {count_parameters(model):,}")
        
        # Move to device
        model.to(device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            test_batch = next(iter(test_loader)).to(device)
            
            if model_name == "ProteinEGNN":
                # EGNN needs positions
                pos = torch.randn(test_batch.x.size(0), 3).to(device)
                output = model(test_batch.x, pos, test_batch.edge_index, test_batch.batch)
            else:
                output = model(test_batch.x, test_batch.edge_index, test_batch.batch)
            
            predictions = torch.softmax(output, dim=1)
            predicted_classes = torch.argmax(output, dim=1)
            
            print(f"  Output shape: {output.shape}")
            print(f"  Sample predictions: {predicted_classes[:5].cpu().numpy()}")
            print(f"  Sample probabilities: {predictions[:2].cpu().numpy()}")
    
    # Quick training example
    print("\n🚀 Quick training example...")
    
    # Use GCN for training
    model = models["ProteinGCN"]
    model.train()
    
    # Create optimizer and loss
    optimizer = create_optimizer(model, "adam", learning_rate=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Train for a few epochs
    for epoch in range(3):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += batch.y.size(0)
            correct += (predicted == batch.y).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        
        print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
    
    # Evaluation
    print("\n📈 Evaluation...")
    
    evaluator = ProteinEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=3
    )
    
    test_loss, test_accuracy, test_metrics = evaluator.test()
    
    print(f"Test Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_accuracy:.4f}")
    print(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    
    # Model comparison
    print("\n🏆 Model Comparison:")
    print("Model           Parameters    Test Accuracy")
    print("-" * 45)
    
    for model_name, model in models.items():
        model.eval()
        evaluator.model = model
        
        with torch.no_grad():
            _, accuracy, _ = evaluator.test()
            param_count = count_parameters(model)
            
            print(f"{model_name:15s} {param_count:8,}    {accuracy:.4f}")
    
    print("\n✅ Example completed successfully!")
    print("\nTo run the full training pipeline:")
    print("  python scripts/train.py")
    print("\nTo launch the interactive demo:")
    print("  streamlit run demo/app.py")


if __name__ == "__main__":
    main()
