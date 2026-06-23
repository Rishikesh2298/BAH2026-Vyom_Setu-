import torch
import torch.nn as nn
import timm

class SpatialViTBranch(nn.Module):
    """
    Vision Transformer branch for spatial feature extraction from high-resolution optical imagery.
    Utilizes a timm-based ViT architecture adapted for 256x256 patches.
    """
    def __init__(self, in_channels: int = 3, embed_dim: int = 768):
        """
        Initializes the ViT Branch.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB or multi-band mock).
            embed_dim (int): The output embedding dimension size.
        """
        super(SpatialViTBranch, self).__init__()
        
        # Using a lightweight ViT for demonstration. In production, larger models might be used.
        # pretrained=False is used here to avoid downloading weights during container build/run,
        # but in a real-world scenario, you'd likely use pretrained=True.
        self.vit = timm.create_model(
            'vit_tiny_patch16_224', 
            pretrained=False, 
            in_chans=in_channels,
            num_classes=0,  # Remove the classification head to extract raw features
            img_size=256    # Force the model to accept 256x256 input
        )
        
        # Assuming vit_tiny outputs 192 dim, we project it to the requested embed_dim
        # vit_tiny feature dim is 192.
        self.feature_proj = nn.Linear(192, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the optical patch.

        Args:
            x (torch.Tensor): Input optical patch of shape (B, in_channels, 256, 256).

        Returns:
            torch.Tensor: Spatial feature representation of shape (B, embed_dim).
        """
        # x is expected to be [Batch, Channels, Height, Width]
        features = self.vit(x)  # Shape: (B, 192)
        projected = self.feature_proj(features)  # Shape: (B, embed_dim)
        return projected
