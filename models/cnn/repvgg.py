import torch
import torch.nn as nn
from models.base import BaseModel

class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Main branch
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride == 1 and in_channels == out_channels:
            self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.conv2 = None
            self.bn2 = None
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        
        if self.conv2 is not None:
            y = y + self.bn2(self.conv2(x))
        
        return y

class RepVGG(BaseModel):
    """RepVGG-A0 implementation optimized for low-resolution inputs."""
    
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__(num_classes)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1
        self.stage1 = nn.Sequential(
            RepVGGBlock(48, 48),
            RepVGGBlock(48, 48),
            RepVGGBlock(48, 48)
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            RepVGGBlock(48, 96, stride=2),
            RepVGGBlock(96, 96),
            RepVGGBlock(96, 96)
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            RepVGGBlock(96, 192, stride=2),
            RepVGGBlock(192, 192),
            RepVGGBlock(192, 192)
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            RepVGGBlock(192, 384, stride=2),
            RepVGGBlock(384, 384),
            RepVGGBlock(384, 384)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(384, num_classes)
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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps from each stage."""
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        x = self.stage1(x)
        features.append(x)
        
        x = self.stage2(x)
        features.append(x)
        
        x = self.stage3(x)
        features.append(x)
        
        x = self.stage4(x)
        features.append(x)
        
        return features 