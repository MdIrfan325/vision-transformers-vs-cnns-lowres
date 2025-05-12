import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
import yaml
import os

def get_augmentation_transforms(config: Dict[str, Any], is_train: bool = True) -> transforms.Compose:
    """Get data augmentation transforms based on configuration."""
    if not is_train or not config['training']['augmentation']['enabled']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) if config['input_channels'] == 1 
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    aug_config = config['training']['augmentation']
    transform_list = []
    
    if aug_config['random_crop']:
        transform_list.append(transforms.RandomCrop(config['img_size'], padding=4))
    
    if aug_config['random_horizontal_flip']:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if aug_config['color_jitter'] and config['input_channels'] == 3:
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    
    if aug_config['random_rotation'] > 0:
        transform_list.append(transforms.RandomRotation(aug_config['random_rotation']))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if config['input_channels'] == 1 
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    return transforms.Compose(transform_list)

def get_dataloader(dataset_name: str, 
                  root_dir: str,
                  batch_size: int,
                  is_train: bool = True,
                  target_size: int = None,
                  config: Dict[str, Any] = None) -> Tuple[DataLoader, int]:
    """Get dataloader for specified dataset."""
    
    # Get dataset configuration
    dataset_config = config['datasets'][dataset_name]
    input_channels = 1 if dataset_name == 'fashion_mnist' else 3
    
    # Create transforms
    transform = get_augmentation_transforms({
        'input_channels': input_channels,
        'img_size': target_size or dataset_config['target_sizes'][0],
        'training': config['training']
    }, is_train)
    
    # Load dataset
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=is_train,
            download=True,
            transform=transform
        )
        num_classes = 10
    elif dataset_name == 'fashion_mnist':
        dataset = torchvision.datasets.FashionMNIST(
            root=root_dir,
            train=is_train,
            download=True,
            transform=transform
        )
        num_classes = 10
    elif dataset_name == 'svhn':
        split = 'train' if is_train else 'test'
        dataset = torchvision.datasets.SVHN(
            root=root_dir,
            split=split,
            download=True,
            transform=transform
        )
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, num_classes

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_dataset_directories(config: Dict[str, Any]) -> None:
    """Create dataset directories if they don't exist."""
    for dataset_name, dataset_config in config['datasets'].items():
        if dataset_config['enabled']:
            os.makedirs(dataset_config['root_dir'], exist_ok=True) 