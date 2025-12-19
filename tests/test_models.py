"""Tests for protein structure prediction models."""

import pytest
import torch
import numpy as np
from torch_geometric.data import Data, Batch

# Add src to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import ProteinGCN, ProteinGAT, ProteinEGNN, ProteinGIN
from src.data import create_synthetic_protein_dataset, ProteinDataProcessor
from src.utils import set_seed, get_device, calculate_rmsd, calculate_mae


class TestModels:
    """Test GNN model implementations."""
    
    def test_protein_gcn(self):
        """Test ProteinGCN model."""
        model = ProteinGCN(input_dim=20, hidden_dim=32, output_dim=3)
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        # Forward pass
        output = model(x, edge_index, batch)
        
        assert output.shape == (1, 3)
        assert not torch.isnan(output).any()
    
    def test_protein_gat(self):
        """Test ProteinGAT model."""
        model = ProteinGAT(input_dim=20, hidden_dim=32, output_dim=3, num_heads=4)
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        # Forward pass
        output = model(x, edge_index, batch)
        
        assert output.shape == (1, 3)
        assert not torch.isnan(output).any()
    
    def test_protein_egnn(self):
        """Test ProteinEGNN model."""
        model = ProteinEGNN(input_dim=20, hidden_dim=32, output_dim=3)
        
        # Create test data
        x = torch.randn(10, 20)
        pos = torch.randn(10, 3)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        # Forward pass
        output = model(x, pos, edge_index, batch)
        
        assert output.shape == (1, 3)
        assert not torch.isnan(output).any()
    
    def test_protein_gin(self):
        """Test ProteinGIN model."""
        model = ProteinGIN(input_dim=20, hidden_dim=32, output_dim=3)
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.randint(0, 10, (2, 20))
        batch = torch.zeros(10, dtype=torch.long)
        
        # Forward pass
        output = model(x, edge_index, batch)
        
        assert output.shape == (1, 3)
        assert not torch.isnan(output).any()


class TestDataProcessing:
    """Test data processing utilities."""
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic protein dataset creation."""
        proteins = create_synthetic_protein_dataset(
            num_proteins=5,
            min_residues=10,
            max_residues=20,
            num_classes=3,
            num_features=20
        )
        
        assert len(proteins) == 5
        
        for protein in proteins:
            assert isinstance(protein, Data)
            assert protein.x.shape[1] == 20  # num_features
            assert protein.y.shape[0] == 1
            assert protein.y.item() < 3  # num_classes
    
    def test_data_processor(self):
        """Test ProteinDataProcessor."""
        processor = ProteinDataProcessor(
            add_positional_encoding=True,
            normalize_features=True,
            add_edge_attributes=True
        )
        
        # Create test data
        x = torch.randn(10, 20)
        edge_index = torch.randint(0, 10, (2, 15))
        data = Data(x=x, edge_index=edge_index)
        
        # Process data
        processed_data = processor(data)
        
        assert processed_data.x.shape[0] == 10
        assert processed_data.x.shape[1] > 20  # Should have positional encoding
        assert processed_data.edge_attr is not None


class TestUtils:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test random seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Set seed again
        set_seed(42)
        
        # Should get same numbers
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_calculate_rmsd(self):
        """Test RMSD calculation."""
        pred_coords = torch.randn(10, 3)
        true_coords = torch.randn(10, 3)
        
        rmsd = calculate_rmsd(pred_coords, true_coords)
        
        assert isinstance(rmsd, torch.Tensor)
        assert rmsd.item() >= 0
    
    def test_calculate_mae(self):
        """Test MAE calculation."""
        pred_values = torch.randn(10)
        true_values = torch.randn(10)
        
        mae = calculate_mae(pred_values, true_values)
        
        assert isinstance(mae, torch.Tensor)
        assert mae.item() >= 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline."""
        # Create synthetic data
        proteins = create_synthetic_protein_dataset(
            num_proteins=10,
            min_residues=5,
            max_residues=10,
            num_classes=2,
            num_features=20
        )
        
        # Create model
        model = ProteinGCN(input_dim=20, hidden_dim=16, output_dim=2)
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        model.train()
        for protein in proteins[:5]:  # Use first 5 for training
            optimizer.zero_grad()
            
            # Create batch
            batch = Batch.from_data_list([protein])
            
            # Forward pass
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(output, batch.y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Test inference
        model.eval()
        with torch.no_grad():
            test_protein = proteins[5]
            batch = Batch.from_data_list([test_protein])
            output = model(batch.x, batch.edge_index, batch.batch)
            
            assert output.shape == (1, 2)
            assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
