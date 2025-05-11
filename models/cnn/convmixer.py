import torch
import torch.nn as nn
from models.base import BaseModel

class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size=9):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                 padding=kernel_size//2, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)
    
    def forward(self, x):
        x = x + self.bn(self.pointwise(self.act(self.depthwise(x))))
        return x

class ConvMixer(BaseModel):
    """ConvMixer implementation optimized for low-resolution inputs."""
    
    def __init__(self, num_classes=10, input_channels=3, dim=256, depth=8, kernel_size=9):
        super().__init__(num_classes)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dim, kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        
        # ConvMixer blocks
        self.blocks = nn.Sequential(*[
            ConvMixerBlock(dim=dim, kernel_size=kernel_size)
            for _ in range(depth)
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dim, num_classes)
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
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps from each stage."""
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 2 == 0:  # Save every other block's output
                features.append(x)
        
        return features 