import torch
import torch.nn as nn

class RadarSpatioTemporalBranch(nn.Module):
    """
    Spatio-Temporal Branch utilizing Conv2D for spatial feature extraction 
    and an LSTM for temporal/multi-angle sequence processing of DFSAR radar patches.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 256, lstm_layers: int = 2, embed_dim: int = 768):
        """
        Initializes the LSTM Branch.

        Args:
            in_channels (int): Number of input channels for a single radar patch.
            hidden_dim (int): Hidden dimension size for the LSTM.
            lstm_layers (int): Number of stacked LSTM layers.
            embed_dim (int): Final output embedding dimension to match or concatenate with the ViT branch.
        """
        super(RadarSpatioTemporalBranch, self).__init__()
        
        # Simple spatial feature extractor for each radar patch
        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1), # 128x128
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 32x32
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # 1x1
            nn.Flatten() # Outputs 128-dim vector per spatial patch
        )
        
        # LSTM to process the sequence of spatial features
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Project LSTM final hidden state to the desired embedding dimension
        self.feature_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the radar sequence patch.

        Args:
            x (torch.Tensor): Input radar sequence of shape (B, Sequence_Length, C, H, W).
                              For a single non-sequential patch, Sequence_Length = 1.

        Returns:
            torch.Tensor: Spatio-temporal feature representation of shape (B, embed_dim).
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Fold the sequence dimension into the batch dimension to process spatially
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.spatial_extractor(x_reshaped)
        
        # Unfold back to (B, Seq_Len, Features)
        spatial_features = spatial_features.view(batch_size, seq_len, -1)
        
        # Process through LSTM
        lstm_out, (hn, cn) = self.lstm(spatial_features)
        
        # Take the hidden state of the last time step from the top LSTM layer
        last_hidden = hn[-1] # Shape: (B, hidden_dim)
        
        # Project to target embedding dimension
        projected = self.feature_proj(last_hidden) # Shape: (B, embed_dim)
        return projected
