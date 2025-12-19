"""Interactive demo for protein structure prediction with GNNs."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models import ProteinGCN, ProteinGAT, ProteinEGNN, ProteinGIN
from src.data import create_synthetic_protein_dataset
from src.utils import get_device


def load_model(model_type: str, input_dim: int, output_dim: int):
    """Load a pre-trained model (or create a random one for demo)."""
    if model_type == "ProteinGCN":
        model = ProteinGCN(input_dim, 64, output_dim)
    elif model_type == "ProteinGAT":
        model = ProteinGAT(input_dim, 64, output_dim, num_heads=4)
    elif model_type == "ProteinEGNN":
        model = ProteinEGNN(input_dim, 64, output_dim)
    elif model_type == "ProteinGIN":
        model = ProteinGIN(input_dim, 64, output_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # For demo purposes, we'll use random weights
    # In practice, you would load from a checkpoint
    model.eval()
    return model


def create_protein_graph(num_residues: int = 50):
    """Create a synthetic protein graph for visualization."""
    # Generate random node features (amino acid types)
    node_features = np.random.randn(num_residues, 20)
    
    # Generate random edges (simulating protein structure)
    num_edges = np.random.randint(num_residues, num_residues * 2)
    edge_list = []
    
    for _ in range(num_edges):
        src = np.random.randint(0, num_residues)
        dst = np.random.randint(0, num_residues)
        if src != dst:  # No self-loops
            edge_list.append((src, dst))
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(num_residues))
    G.add_edges_from(edge_list)
    
    return G, node_features


def visualize_protein_graph(G, node_features, predictions=None):
    """Visualize protein graph with predictions."""
    # Create 3D layout
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Extract coordinates
    x_coords = [pos[node][0] for node in G.nodes()]
    y_coords = [pos[node][1] for node in G.nodes()]
    z_coords = [pos[node][2] for node in G.nodes()]
    
    # Create edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    # Create traces
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Node colors based on predictions or features
    if predictions is not None:
        node_colors = predictions
        colorbar_title = "Prediction Score"
    else:
        node_colors = np.sum(node_features, axis=1)
        colorbar_title = "Feature Sum"
    
    node_trace = go.Scatter3d(
        x=x_coords, y=y_coords, z=z_coords,
        mode='markers',
        marker=dict(
            size=8,
            color=node_colors,
            colorscale='Viridis',
            colorbar=dict(title=colorbar_title),
            line=dict(width=2, color='black')
        ),
        text=[f"Residue {i}" for i in G.nodes()],
        hovertemplate='%{text}<br>Score: %{marker.color}<extra></extra>'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title="Protein Structure Graph",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="cube"
        ),
        width=800,
        height=600
    )
    
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Protein Structure Prediction with GNNs",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Protein Structure Prediction with Graph Neural Networks")
    st.markdown("Interactive demo for exploring protein structure prediction using various GNN architectures.")
    
    # Sidebar controls
    st.sidebar.header("Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["ProteinGCN", "ProteinGAT", "ProteinEGNN", "ProteinGIN"],
        help="Choose the GNN architecture to use"
    )
    
    num_residues = st.sidebar.slider(
        "Number of Residues",
        min_value=20,
        max_value=200,
        value=50,
        help="Number of amino acid residues in the protein"
    )
    
    num_classes = st.sidebar.selectbox(
        "Number of Classes",
        [2, 3, 4, 5],
        value=3,
        help="Number of protein classes for classification"
    )
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_type, 20, num_classes)
        device = get_device()
        model.to(device)
    
    st.sidebar.success(f"Model loaded: {model_type}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Protein Structure Visualization")
        
        # Generate protein graph
        if st.button("Generate New Protein", type="primary"):
            st.session_state.protein_graph = create_protein_graph(num_residues)
        
        if 'protein_graph' not in st.session_state:
            st.session_state.protein_graph = create_protein_graph(num_residues)
        
        G, node_features = st.session_state.protein_graph
        
        # Make predictions
        with torch.no_grad():
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0)
            edge_index = torch.tensor(list(G.edges()), dtype=torch.long).t().contiguous()
            batch = torch.zeros(x.size(1), dtype=torch.long)
            
            # Add reverse edges for undirected graph
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            
            # Move to device
            x = x.to(device)
            edge_index = edge_index.to(device)
            batch = batch.to(device)
            
            # Make prediction
            if model_type == "ProteinEGNN":
                # For EGNN, we need positions
                pos = torch.randn(x.size(1), 3).to(device)
                output = model(x.squeeze(0), pos, edge_index, batch)
            else:
                output = model(x.squeeze(0), edge_index, batch)
            
            # Get predictions
            predictions = torch.softmax(output, dim=1).squeeze().cpu().numpy()
            predicted_class = torch.argmax(output, dim=1).item()
        
        # Visualize graph
        fig = visualize_protein_graph(G, node_features, predictions)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("Model Predictions")
        
        # Display predictions
        st.subheader("Class Probabilities")
        for i, prob in enumerate(predictions):
            st.metric(f"Class {i}", f"{prob:.3f}")
        
        st.subheader("Prediction Summary")
        st.metric("Predicted Class", predicted_class)
        st.metric("Confidence", f"{max(predictions):.3f}")
        
        # Model information
        st.subheader("Model Information")
        st.info(f"**Model Type:** {model_type}")
        st.info(f"**Parameters:** {sum(p.numel() for p in model.parameters()):,}")
        st.info(f"**Device:** {device}")
        
        # Graph statistics
        st.subheader("Graph Statistics")
        st.metric("Nodes", G.number_of_nodes())
        st.metric("Edges", G.number_of_edges())
        st.metric("Density", f"{nx.density(G):.3f}")
    
    # Additional analysis
    st.header("Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Node Feature Distribution")
        feature_sums = np.sum(node_features, axis=1)
        
        fig_hist = px.histogram(
            x=feature_sums,
            nbins=20,
            title="Distribution of Node Feature Sums",
            labels={"x": "Feature Sum", "y": "Count"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col4:
        st.subheader("Degree Distribution")
        degrees = [G.degree(n) for n in G.nodes()]
        
        fig_degree = px.histogram(
            x=degrees,
            nbins=20,
            title="Node Degree Distribution",
            labels={"x": "Degree", "y": "Count"}
        )
        st.plotly_chart(fig_degree, use_container_width=True)
    
    # Model comparison
    st.header("Model Comparison")
    
    if st.button("Compare All Models"):
        comparison_results = {}
        
        for model_name in ["ProteinGCN", "ProteinGAT", "ProteinEGNN", "ProteinGIN"]:
            with torch.no_grad():
                comp_model = load_model(model_name, 20, num_classes)
                comp_model.to(device)
                comp_model.eval()
                
                if model_name == "ProteinEGNN":
                    pos = torch.randn(x.size(1), 3).to(device)
                    comp_output = comp_model(x.squeeze(0), pos, edge_index, batch)
                else:
                    comp_output = comp_model(x.squeeze(0), edge_index, batch)
                
                comp_predictions = torch.softmax(comp_output, dim=1).squeeze().cpu().numpy()
                comp_predicted_class = torch.argmax(comp_output, dim=1).item()
                
                comparison_results[model_name] = {
                    "predicted_class": comp_predicted_class,
                    "confidence": max(comp_predictions),
                    "probabilities": comp_predictions
                }
        
        # Display comparison
        st.subheader("Model Predictions Comparison")
        
        comparison_df = []
        for model_name, results in comparison_results.items():
            comparison_df.append({
                "Model": model_name,
                "Predicted Class": results["predicted_class"],
                "Confidence": f"{results['confidence']:.3f}"
            })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Probability comparison
        st.subheader("Probability Comparison")
        prob_df = []
        for model_name, results in comparison_results.items():
            for i, prob in enumerate(results["probabilities"]):
                prob_df.append({
                    "Model": model_name,
                    "Class": i,
                    "Probability": prob
                })
        
        fig_comparison = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            color="Model",
            title="Class Probabilities by Model",
            barmode="group"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "This demo showcases various Graph Neural Network architectures for protein structure prediction. "
        "The models are trained on synthetic data for demonstration purposes. "
        "In practice, these models would be trained on real protein datasets like PDB or AlphaFold."
    )


if __name__ == "__main__":
    main()
