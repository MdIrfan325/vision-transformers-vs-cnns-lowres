import os
import torch
import torchvision
import torchvision.transforms as transforms
from pathlib import Path
import yaml
from tqdm import tqdm

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_cifar10(root_dir: str) -> None:
    """Download and prepare CIFAR-10 dataset."""
    print("Downloading CIFAR-10 dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download training set
    trainset = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test set
    testset = torchvision.datasets.CIFAR10(
        root=root_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"CIFAR-10 dataset downloaded to {root_dir}")
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")

def download_fashion_mnist(root_dir: str) -> None:
    """Download and prepare Fashion-MNIST dataset."""
    print("Downloading Fashion-MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Download training set
    trainset = torchvision.datasets.FashionMNIST(
        root=root_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    # Download test set
    testset = torchvision.datasets.FashionMNIST(
        root=root_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"Fashion-MNIST dataset downloaded to {root_dir}")
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")

def download_svhn(root_dir: str) -> None:
    """Download and prepare SVHN dataset."""
    print("Downloading SVHN dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download training set
    trainset = torchvision.datasets.SVHN(
        root=root_dir,
        split='train',
        download=True,
        transform=transform
    )
    
    # Download test set
    testset = torchvision.datasets.SVHN(
        root=root_dir,
        split='test',
        download=True,
        transform=transform
    )
    
    print(f"SVHN dataset downloaded to {root_dir}")
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")

def main():
    # Load configuration
    config = load_config('configs/experiment_config.yaml')
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Download datasets based on configuration
    for dataset_name, dataset_config in config['datasets'].items():
        if not dataset_config['enabled']:
            continue
            
        print(f"\nProcessing {dataset_name}...")
        dataset_dir = Path(dataset_config['root_dir'])
        dataset_dir.mkdir(exist_ok=True)
        
        if dataset_name == 'cifar10':
            download_cifar10(str(dataset_dir))
        elif dataset_name == 'fashion_mnist':
            download_fashion_mnist(str(dataset_dir))
        elif dataset_name == 'svhn':
            download_svhn(str(dataset_dir))
        else:
            print(f"Unknown dataset: {dataset_name}")
    
    print("\nAll datasets downloaded successfully!")

if __name__ == '__main__':
    main() 