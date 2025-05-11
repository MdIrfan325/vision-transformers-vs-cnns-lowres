import torch
import torch.nn as nn
from models.base import BaseModel

class LeNet5(BaseModel):
    """Modified LeNet-5 architecture for low-resolution inputs."""
    
    def __init__(self, num_classes=10, input_channels=3):
        super().__init__(num_classes)
        
        self.features = nn.Sequential(
            # First conv layer
            nn.Conv2d(input_channels, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv layer
            nn.Conv2d(6, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv layer (added for better feature extraction)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Calculate the size of the flattened features
        self._to_linear = None
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def _get_conv_output_size(self, x):
        """Calculate the size of the flattened features."""
        x = self.features(x)
        return x.size(1) * x.size(2) * x.size(3)
    
    def forward(self, x):
        if self._to_linear is None:
            self._to_linear = self._get_conv_output_size(x)
            self.classifier[0] = nn.Linear(self._to_linear, 64)
        
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_feature_maps(self, x):
        """Get feature maps from the last convolutional layer."""
        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
        return features[-1]  # Return the last feature map 