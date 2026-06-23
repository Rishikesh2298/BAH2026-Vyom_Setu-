import torch
import torch.nn as nn
from typing import Tuple

from .vit_branch import SpatialViTBranch
from .lstm_branch import RadarSpatioTemporalBranch

class LUNANetFusion(nn.Module):
    """
    Cross-modal fusion network combining optical (ViT) and radar (LSTM) features.
    Outputs to three multi-task heads: Hazard Segmentation, Water-Ice Regression, and Surface Roughness.
    """
    def __init__(self, 
                 optical_channels: int = 3, 
                 radar_channels: int = 1, 
                 embed_dim: int = 512):
        """
        Initializes the LUNA-Net dual-branch fusion architecture.

        Args:
            optical_channels (int): Input channels for the optical branch.
            radar_channels (int): Input channels for the radar branch.
            embed_dim (int): Common embedding dimension for both branches.
        """
        super(LUNANetFusion, self).__init__()
        
        self.vit_branch = SpatialViTBranch(in_channels=optical_channels, embed_dim=embed_dim)
        self.lstm_branch = RadarSpatioTemporalBranch(in_channels=radar_channels, embed_dim=embed_dim)
        
        # Feature fusion via simple concatenation and dense projection
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Head 1: Hazard Segmentation Head
        # Normally this would be a full U-Net decoder using intermediate branch features.
        # Since ViT/LSTM return condensed 1D embeddings, we'll build a small transposed CNN decoder
        # to upsample the fused embedding back to a 256x256 mask.
        self.segmentation_head = nn.Sequential(
            nn.Linear(embed_dim, 256 * 8 * 8), # Expand to 8x8 spatial grid
            nn.GELU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 128x128
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),    # 256x256
            nn.Sigmoid() # Binary mask
        )
        
        # Head 2: Subsurface Water-Ice Regression Head (Output 0.0 to 1.0)
        self.water_ice_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Head 3: Surface Roughness Head (Continuous scalar)
        self.roughness_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

    def forward(self, optical_input: torch.Tensor, radar_seq_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the full LUNA-Net architecture.

        Args:
            optical_input (torch.Tensor): Optical patch (B, C_opt, 256, 256).
            radar_seq_input (torch.Tensor): Radar sequence (B, Seq_Len, C_rad, 256, 256).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Hazard segmentation mask (B, 1, 256, 256)
                - Water-ice probability (B, 1)
                - Surface roughness index (B, 1)
        """
        # Extract features
        opt_features = self.vit_branch(optical_input) # (B, embed_dim)
        rad_features = self.lstm_branch(radar_seq_input) # (B, embed_dim)
        
        # Concatenate and fuse
        fused = torch.cat([opt_features, rad_features], dim=-1) # (B, 2 * embed_dim)
        fused_embedding = self.fusion_layer(fused) # (B, embed_dim)
        
        # Pass through heads
        seg_mask = self.segmentation_head(fused_embedding)
        water_ice_prob = self.water_ice_head(fused_embedding)
        roughness_idx = self.roughness_head(fused_embedding)
        
        return seg_mask, water_ice_prob, roughness_idx
