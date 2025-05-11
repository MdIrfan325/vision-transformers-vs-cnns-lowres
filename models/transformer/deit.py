import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from models.base import BaseModel

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class DeiT(BaseModel):
    """DeiT-Tiny implementation optimized for low-resolution inputs."""
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., distilled=False):
        super().__init__(num_classes)
        
        # Patch embedding
        self.patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                     p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        num_patches = (img_size // patch_size) ** 2
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))  # +2 for cls and dist tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        
        # Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classifier
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        B = x.shape[0]
        
        # Add class token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add distillation token if enabled
        if self.dist_token is not None:
            dist_token = repeat(self.dist_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([x[:, :1], dist_token, x[:, 1:]], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        attn_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attn_weights.append(attn)
        
        x = self.norm(x)
        
        # Use only class token for classification
        x = x[:, 0]
        x = self.head(x)
        
        return x
    
    def get_attention_maps(self, x):
        """Get attention maps from all transformer blocks."""
        # Patch embedding
        x = self.patch_embed(x)
        B = x.shape[0]
        
        # Add class token
        cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=B)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add distillation token if enabled
        if self.dist_token is not None:
            dist_token = repeat(self.dist_token, '1 1 d -> b 1 d', b=B)
            x = torch.cat([x[:, :1], dist_token, x[:, 1:]], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Get attention maps from each block
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)
        
        return attn_maps 