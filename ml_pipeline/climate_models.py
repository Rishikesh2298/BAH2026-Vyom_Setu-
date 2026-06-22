"""
Neural Network Models for Climate Forecasting
LSTM, CNN, and Transformer-based surrogate models
"""

import logging
import torch
import torch.nn as nn
from typing import Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)

class LSTMClimateForecaster(nn.Module):
    """
    LSTM-based Climate Forecaster
    Captures temporal dynamics in climate variables
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256, 
                 num_layers: int = 3, output_size: int = 1,
                 dropout: float = 0.2):
        """
        Initialize LSTM forecaster
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden layer dimension
            num_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
        """
        super(LSTMClimateForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Dense layers for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        logger.info(f"Initialized LSTM Forecaster: input={input_size}, hidden={hidden_size}, output={output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # LSTM pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Dense layers
        output = self.fc(last_hidden)
        
        return output

class ConvolutionalClimateModel(nn.Module):
    """
    CNN-based Climate Model
    Captures spatial patterns in climate data
    """
    
    def __init__(self, in_channels: int = 1, num_filters: int = 32,
                 kernel_size: int = 3):
        """
        Initialize CNN climate model
        
        Args:
            in_channels: Number of input channels
            num_filters: Number of convolutional filters
            kernel_size: Size of convolution kernel
        """
        super(ConvolutionalClimateModel, self).__init__()
        
        self.in_channels = in_channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters, kernel_size, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Activation
        self.relu = nn.ReLU()
        
        logger.info(f"Initialized CNN Model: in_channels={in_channels}, filters={num_filters}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output feature maps
        """
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Conv block 3
        x = self.relu(self.conv3(x))
        
        return x

class TransformerClimateForecaster(nn.Module):
    """
    Transformer-based Climate Forecaster
    Attention-based model for temporal sequences
    """
    
    def __init__(self, input_size: int, d_model: int = 256,
                 nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 1024, dropout: float = 0.1):
        """
        Initialize Transformer forecaster
        
        Args:
            input_size: Number of input features
            d_model: Dimension of model
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerClimateForecaster, self).__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, d_model)  # Max sequence length 100
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, input_size)
        
        logger.info(f"Initialized Transformer Forecaster: d_model={d_model}, nhead={nhead}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, seq_length, input_size)
        """
        # Embedding
        x = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Output
        x = self.output_layer(x)
        
        return x

class HybridClimateModel(nn.Module):
    """
    Hybrid model combining CNN for spatial patterns and LSTM for temporal dynamics
    """
    
    def __init__(self, spatial_channels: int = 1, temporal_features: int = 50,
                 hidden_size: int = 128, num_filters: int = 32):
        """
        Initialize hybrid climate model
        
        Args:
            spatial_channels: Number of input channels for CNN
            temporal_features: Features from CNN for LSTM
            hidden_size: LSTM hidden size
            num_filters: Number of CNN filters
        """
        super(HybridClimateModel, self).__init__()
        
        # Spatial extractor (CNN)
        self.spatial_branch = ConvolutionalClimateModel(
            in_channels=spatial_channels,
            num_filters=num_filters
        )
        
        # Temporal predictor (LSTM)
        self.temporal_branch = LSTMClimateForecaster(
            input_size=temporal_features,
            hidden_size=hidden_size,
            output_size=1
        )
        
        logger.info("Initialized Hybrid CNN-LSTM Climate Model")
    
    def forward(self, spatial_input: torch.Tensor, 
                temporal_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            spatial_input: Spatial field (batch, channels, height, width)
            temporal_input: Temporal sequence (batch, seq_len, features)
            
        Returns:
            Tuple of (spatial_features, temporal_prediction)
        """
        # Spatial branch
        spatial_features = self.spatial_branch(spatial_input)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)
        
        # Temporal branch
        temporal_prediction = self.temporal_branch(temporal_input)
        
        return spatial_features, temporal_prediction

class GraphNeuralNetworkModel(nn.Module):
    """
    Graph Neural Network for regional climate interactions
    Models spatial dependencies in climate variables
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 output_dim: int = 1, num_layers: int = 3):
        """
        Initialize GNN
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            num_layers: Number of graph layers
        """
        super(GraphNeuralNetworkModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers (placeholder)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
        logger.info(f"Initialized GNN: input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            node_features: Node feature matrix (num_nodes, input_dim)
            adjacency_matrix: Graph adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Node predictions (num_nodes, output_dim)
        """
        x = node_features
        
        for layer in self.layers:
            x = layer(x)
            x = self.relu(x)
            # Graph aggregation (simplified)
            x = torch.mm(adjacency_matrix, x)
        
        output = self.output_layer(x)
        return output

def create_model(model_type: str, config: dict) -> nn.Module:
    """
    Factory function to create climate models
    
    Args:
        model_type: Type of model ('lstm', 'cnn', 'transformer', 'hybrid', 'gnn')
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized model
    """
    logger.info(f"Creating {model_type} model with config: {config}")
    
    if model_type == 'lstm':
        return LSTMClimateForecaster(**config)
    elif model_type == 'cnn':
        return ConvolutionalClimateModel(**config)
    elif model_type == 'transformer':
        return TransformerClimateForecaster(**config)
    elif model_type == 'hybrid':
        return HybridClimateModel(**config)
    elif model_type == 'gnn':
        return GraphNeuralNetworkModel(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
