import torch
import torch.nn as nn
from einops import rearrange
from models.base import BaseModel

class ConvEmbed(nn.Module):
    """Convolutional Embedding Layer."""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class ConvAttention(nn.Module):
    """Convolutional Attention Layer."""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ConvBlock(nn.Module):
    """Convolutional Transformer Block."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = ConvAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Stage(nn.Module):
    """Stage of CvT."""
    def __init__(self, dim, depth, num_heads, mlp_ratio, qkv_bias, drop, attn_drop):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop)
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class CvT(BaseModel):
    """Convolutional vision Transformer implementation optimized for low-resolution inputs."""
    
    def __init__(self, num_classes=10, in_chans=3, img_size=32,
                 embed_dims=[64, 192, 384], depths=[1, 4, 3],
                 num_heads=[1, 3, 6], mlp_ratios=[4, 4, 4],
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0.):
        super().__init__(num_classes)
        
        self.num_stages = len(embed_dims)
        
        # Patch Embedding
        self.patch_embed = nn.ModuleList()
        self.pos_embed = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        # Stage 1
        self.patch_embed.append(ConvEmbed(in_chans, embed_dims[0], 7, 4, 2))
        self.pos_embed.append(nn.Parameter(torch.zeros(1, embed_dims[0], img_size//4, img_size//4)))
        self.stages.append(Stage(embed_dims[0], depths[0], num_heads[0],
                               mlp_ratios[0], qkv_bias, drop_rate, attn_drop_rate))
        
        # Stage 2
        self.patch_embed.append(ConvEmbed(embed_dims[0], embed_dims[1], 3, 2, 1))
        self.pos_embed.append(nn.Parameter(torch.zeros(1, embed_dims[1], img_size//8, img_size//8)))
        self.stages.append(Stage(embed_dims[1], depths[1], num_heads[1],
                               mlp_ratios[1], qkv_bias, drop_rate, attn_drop_rate))
        
        # Stage 3
        self.patch_embed.append(ConvEmbed(embed_dims[1], embed_dims[2], 3, 2, 1))
        self.pos_embed.append(nn.Parameter(torch.zeros(1, embed_dims[2], img_size//16, img_size//16)))
        self.stages.append(Stage(embed_dims[2], depths[2], num_heads[2],
                               mlp_ratios[2], qkv_bias, drop_rate, attn_drop_rate))
        
        # Head
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(self.num_stages):
            x = self.patch_embed[i](x)
            x = x + self.pos_embed[i]
            x = self.stages[i](x)
        
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps from each stage."""
        features = []
        
        for i in range(self.num_stages):
            x = self.patch_embed[i](x)
            x = x + self.pos_embed[i]
            x = self.stages[i](x)
            features.append(x)
        
        return features 