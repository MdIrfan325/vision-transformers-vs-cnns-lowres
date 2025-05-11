import torch
import torch.nn as nn
from einops import rearrange
from models.base import BaseModel

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        
        # Local representation
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size//2, groups=channel)
        self.conv2 = nn.Conv2d(channel, dim, 1)
        
        # Global representation
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # Fusion
        self.conv3 = nn.Conv2d(dim, channel, 1)
        self.conv4 = nn.Conv2d(2 * channel, channel, kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        
        # Reshape for transformer
        B, C, H, W = y.shape
        y = rearrange(y, 'b c (h ph) (w pw) -> b (h w) (ph pw c)',
                     ph=self.ph, pw=self.pw)
        
        # Transformer
        y = self.transformer(y)
        
        # Reshape back
        y = rearrange(y, 'b (h w) (ph pw c) -> b c (h ph) (w pw)',
                     h=H//self.ph, w=W//self.pw, ph=self.ph, pw=self.pw)
        
        # Fusion
        y = self.conv3(y)
        y = torch.cat([x, y], dim=1)
        y = self.conv4(y)
        
        return y

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 1, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class MobileViT(BaseModel):
    """MobileViT implementation optimized for low-resolution inputs."""
    
    def __init__(self, num_classes=10, input_channels=3, dims=[64, 80, 96],
                 channels=[16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
                 kernel_size=3, patch_size=(2, 2), depth=2, mlp_dim=256):
        super().__init__(num_classes)
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU()
        )
        
        # MobileViT blocks
        self.blocks = nn.ModuleList()
        for i in range(len(dims)):
            self.blocks.append(MobileViTBlock(
                dim=dims[i],
                depth=depth,
                channel=channels[i+1],
                kernel_size=kernel_size,
                patch_size=patch_size,
                mlp_dim=mlp_dim
            ))
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(channels[-1], channels[-1], kernel_size=1),
            nn.BatchNorm2d(channels[-1]),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], num_classes)
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
        
        for block in self.blocks:
            x = block(x)
        
        x = self.head(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps from each stage."""
        features = []
        
        x = self.stem(x)
        features.append(x)
        
        for block in self.blocks:
            x = block(x)
            features.append(x)
        
        return features 