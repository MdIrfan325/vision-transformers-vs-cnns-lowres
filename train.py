import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import yaml
from tqdm import tqdm
import wandb
from pathlib import Path
import importlib
import sys

from data.datasets import get_dataloader, load_config, prepare_dataset_directories

def get_model(model_name: str, model_config: dict, num_classes: int) -> nn.Module:
    """Dynamically import and instantiate model."""
    model_type = model_config['type']
    model_module = importlib.import_module(f'models.{model_type}.{model_name}')
    
    # Handle special cases for model class names
    model_class_map = {
        'lenet5': 'LeNet5',
        'resnet18': 'ResNet18',
        'efficientnetv2_s': 'EfficientNetV2S',
        'mobilenetv2': 'MobileNetV2',
        'convmixer': 'ConvMixer',
        'repvgg': 'RepVGG',
        'vit': 'ViT',
        'deit': 'DeiT',
        'mobilevit': 'MobileViT',
        'swin': 'SwinTransformer',
        'cvt': 'CvT'
    }
    
    model_class_name = model_class_map.get(model_name, model_name.replace('-', '').title())
    model_class = getattr(model_module, model_class_name)
    
    # Remove num_classes from params if it exists to avoid duplicate argument
    params = model_config['params'].copy()
    if 'num_classes' in params:
        del params['num_classes']
    
    return model_class(num_classes=num_classes, **params)

def train_epoch(model: nn.Module, 
                train_loader: torch.utils.data.DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model: nn.Module,
            val_loader: torch.utils.data.DataLoader,
            criterion: nn.Module,
            device: torch.device) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    # Load configuration
    config = load_config('configs/experiment_config.yaml')
    
    # Create directories
    prepare_dataset_directories(config)
    
    # Initialize wandb
    if config['logging']['wandb']['enabled']:
        wandb.init(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb']['entity'],
            config=config
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training loop for each dataset and model
    for dataset_name, dataset_config in config['datasets'].items():
        if not dataset_config['enabled']:
            continue
            
        for target_size in dataset_config['target_sizes']:
            # Get dataloaders
            train_loader, num_classes = get_dataloader(
                dataset_name=dataset_name,
                root_dir=dataset_config['root_dir'],
                batch_size=config['training']['batch_size'],
                is_train=True,
                target_size=target_size,
                config=config
            )
            
            val_loader, _ = get_dataloader(
                dataset_name=dataset_name,
                root_dir=dataset_config['root_dir'],
                batch_size=config['training']['batch_size'],
                is_train=False,
                target_size=target_size,
                config=config
            )
            
            # Train each model
            for model_name, model_config in config['models'].items():
                if not model_config['enabled']:
                    continue
                
                print(f"\nTraining {model_name} on {dataset_name} at {target_size}x{target_size}")
                
                # Initialize model
                model = get_model(model_name, model_config, num_classes)
                model = model.to(device)
                
                # Initialize optimizer and scheduler
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config['training']['learning_rate'],
                    weight_decay=config['training']['weight_decay']
                )
                
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=config['training']['scheduler']['warmup_epochs'],
                    T_mult=2
                )
                
                criterion = nn.CrossEntropyLoss()
                
                # Training loop
                best_acc = 0
                for epoch in range(config['training']['num_epochs']):
                    # Train
                    train_loss, train_acc = train_epoch(
                        model, train_loader, criterion, optimizer, device, epoch
                    )
                    
                    # Validate
                    val_loss, val_acc = validate(
                        model, val_loader, criterion, device
                    )
                    
                    # Update scheduler
                    scheduler.step()
                    
                    # Log metrics
                    if config['logging']['wandb']['enabled']:
                        wandb.log({
                            f'{model_name}_{dataset_name}_{target_size}/train_loss': train_loss,
                            f'{model_name}_{dataset_name}_{target_size}/train_acc': train_acc,
                            f'{model_name}_{dataset_name}_{target_size}/val_loss': val_loss,
                            f'{model_name}_{dataset_name}_{target_size}/val_acc': val_acc,
                            f'{model_name}_{dataset_name}_{target_size}/epoch': epoch
                        })
                    
                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc
                        save_dir = Path(config['logging']['save_dir'])
                        save_dir.mkdir(exist_ok=True)
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'val_acc': val_acc
                        }, save_dir / f'{model_name}_{dataset_name}_{target_size}.pth')
                
                print(f"Best validation accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 