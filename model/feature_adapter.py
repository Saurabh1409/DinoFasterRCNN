"""
Feature Adapter for frozen DINOv2 backbone.
Adds trainable parameters to adapt backbone features for object detection.
"""
import torch
import torch.nn as nn
from typing import Dict


class FeatureAdapter(nn.Module):
    """
    Lightweight adapter that transforms frozen backbone features.
    Uses bottleneck architecture for efficiency.
    """
    
    def __init__(self, in_channels=768, adapter_dim=256, num_levels=4):
        """
        Args:
            in_channels: Input feature channels from backbone (DINOv2-base = 768)
            adapter_dim: Adapter bottleneck dimension (smaller = fewer params)
            num_levels: Number of feature pyramid levels
        """
        super().__init__()
        self.in_channels = in_channels
        self.adapter_dim = adapter_dim
        self.num_levels = num_levels
        
        # Adapter modules for each feature level
        self.adapters = nn.ModuleDict()
        
        for i in range(num_levels):
            # Bottleneck: down -> adapt -> up
            # This follows AdaptFormer/LoRA-style bottleneck
            self.adapters[f'adapter_{i}'] = nn.Sequential(
                # Down-projection (reduce channels)
                nn.Conv2d(in_channels, adapter_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(adapter_dim),
                nn.ReLU(inplace=True),
                
                # Adaptation (3x3 conv for spatial reasoning)
                nn.Conv2d(adapter_dim, adapter_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(adapter_dim),
                nn.ReLU(inplace=True),
                
                # Up-projection (restore channels)
                nn.Conv2d(adapter_dim, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
            )
        
        # Residual scaling factor (similar to LoRA)
        self.scale = 0.5  # Start with small residual connection
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply adapters to backbone features.
        
        Args:
            features: Dict of feature maps {'0': tensor, '1': tensor, ...}
        
        Returns:
            Adapted features with same structure
        """
        adapted_features = {}
        
        for level, feat_map in features.items():
            adapter = self.adapters[f'adapter_{int(level)}']
            
            # Residual connection: adapted_feat = feat + scale * adapter(feat)
            adapted = adapter(feat_map)
            adapted_features[level] = feat_map + self.scale * adapted
        
        return adapted_features


class SimpleFeatureAdapter(nn.Module):
    """
    Even simpler adapter - just 1x1 conv + residual.
    Fewer parameters, faster training.
    """
    
    def __init__(self, in_channels=768, reduction=4, num_levels=4):
        """
        Args:
            in_channels: Input feature channels
            reduction: Channel reduction factor (adapter_dim = in_channels // reduction)
            num_levels: Number of feature levels
        """
        super().__init__()
        adapter_dim = in_channels // reduction
        
        self.adapters = nn.ModuleDict()
        for i in range(num_levels):
            # Simple: down -> ReLU -> up
            self.adapters[f'adapter_{i}'] = nn.Sequential(
                nn.Conv2d(in_channels, adapter_dim, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(adapter_dim, in_channels, 1),
            )
        
        self.scale = 0.5
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adapted_features = {}
        for level, feat_map in features.items():
            adapter = self.adapters[f'adapter_{int(level)}']
            adapted_features[level] = feat_map + self.scale * adapter(feat_map)
        return adapted_features


class AdapterWrapper(nn.Module):
    """
    Wrapper that adds adapter to frozen backbone.
    """
    
    def __init__(self, backbone, adapter_type='simple', adapter_dim=256):
        """
        Args:
            backbone: Frozen backbone model
            adapter_type: 'simple' or 'feature' (FeatureAdapter)
            adapter_dim: Adapter dimension (only for 'feature' type)
        """
        super().__init__()
        self.backbone = backbone
        self.backbone.eval()  # Ensure backbone stays in eval mode
        
        # Get number of feature levels from backbone
        num_levels = len(backbone.out_indices) if hasattr(backbone, 'out_indices') else 4
        in_channels = backbone.out_channels
        
        # Create adapter
        if adapter_type == 'simple':
            self.adapter = SimpleFeatureAdapter(in_channels, reduction=4, num_levels=num_levels)
        else:
            self.adapter = FeatureAdapter(in_channels, adapter_dim, num_levels)
    
    def forward(self, images):
        # Backbone forward (frozen, no gradients)
        with torch.no_grad():
            features = self.backbone(images)
        
        # Adapter forward (trainable)
        adapted_features = self.adapter(features)
        return adapted_features




