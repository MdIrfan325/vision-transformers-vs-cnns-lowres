import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all models in the experiment."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        
    @abstractmethod
    def forward(self, x):
        pass
    
    def get_flops(self, input_size=(1, 3, 32, 32)):
        """Calculate FLOPs for the model."""
        from thop import profile
        input = torch.randn(input_size)
        flops, params = profile(self, inputs=(input,))
        return flops, params
    
    def get_model_size(self):
        """Calculate model size in MB."""
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb
    
    def get_attention_maps(self, x):
        """Get attention maps if the model supports it."""
        return None
    
    def get_feature_maps(self, x):
        """Get feature maps for visualization."""
        return None 