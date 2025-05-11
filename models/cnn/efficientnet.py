import torch
import torch.nn as nn
from models.base import BaseModel

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = in_channels == out_channels and stride == 1
        
        hidden_dim = int(round(in_channels * expand_ratio))
        
        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)

class EfficientNetV2S(BaseModel):
    """EfficientNetV2-S implementation optimized for low-resolution inputs."""
    
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__(num_classes)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 24, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU()
        )
        
        # MBConv blocks
        self.blocks = nn.Sequential(
            # Stage 1
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=1),
            MBConv(24, 24, kernel_size=3, stride=1, expand_ratio=1),
            
            # Stage 2
            MBConv(24, 48, kernel_size=3, stride=2, expand_ratio=4),
            MBConv(48, 48, kernel_size=3, stride=1, expand_ratio=4),
            MBConv(48, 48, kernel_size=3, stride=1, expand_ratio=4),
            
            # Stage 3
            MBConv(48, 64, kernel_size=3, stride=2, expand_ratio=4),
            MBConv(64, 64, kernel_size=3, stride=1, expand_ratio=4),
            MBConv(64, 64, kernel_size=3, stride=1, expand_ratio=4),
            
            # Stage 4
            MBConv(64, 128, kernel_size=3, stride=2, expand_ratio=4),
            MBConv(128, 128, kernel_size=3, stride=1, expand_ratio=4),
            MBConv(128, 128, kernel_size=3, stride=1, expand_ratio=4),
            MBConv(128, 128, kernel_size=3, stride=1, expand_ratio=4),
            
            # Stage 5
            MBConv(128, 160, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(160, 160, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(160, 160, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(160, 160, kernel_size=3, stride=1, expand_ratio=6),
            
            # Stage 6
            MBConv(160, 272, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(272, 272, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(272, 272, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(272, 272, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(272, 272, kernel_size=3, stride=1, expand_ratio=6),
            
            # Stage 7
            MBConv(272, 448, kernel_size=3, stride=2, expand_ratio=6),
            MBConv(448, 448, kernel_size=3, stride=1, expand_ratio=6),
            MBConv(448, 448, kernel_size=3, stride=1, expand_ratio=6)
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(448, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps from each stage."""
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [2, 5, 9, 13, 18]:  # End of each stage
                features.append(x)
        
        return features 