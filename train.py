import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import wandb
from tqdm import tqdm
import time
from pathlib import Path

from data.datasets import get_dataloader, get_augmentation_transforms
from models.cnn.lenet5 import LeNet5
from models.cnn.resnet import ModifiedResNet18
from models.transformer.vit import VisionTransformer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(model_name, config, num_classes=10):
    """Initialize model based on configuration."""
    if model_name == 'lenet5':
        return LeNet5(num_classes=num_classes)
    elif model_name == 'resnet18':
        return ModifiedResNet18(
            num_classes=num_classes,
            stem_kernel_size=config['models']['resnet18']['stem_kernel_size'],
            stem_stride=config['models']['resnet18']['stem_stride']
        )
    elif model_name == 'vit_tiny':
        return VisionTransformer(
            img_size=config['datasets']['cifar10']['original_size'],
            patch_size=config['models']['vit_tiny']['patch_size'],
            embed_dim=config['models']['vit_tiny']['embed_dim'],
            depth=config['models']['vit_tiny']['depth'],
            num_heads=config['models']['vit_tiny']['num_heads'],
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=config['training']['mixed_precision']):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        if config['training']['mixed_precision']:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total

def main():
    # Load configuration
    config = load_config('configs/experiment_config.yaml')
    
    # Initialize wandb
    if config['logging']['wandb']['enabled']:
        wandb.init(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb']['entity'],
            config=config
        )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create results directory
    results_dir = Path(config['logging']['save_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Train and evaluate each model
    for model_name, model_config in config['models'].items():
        if not model_config['enabled']:
            continue
        
        print(f"\nTraining {model_name}...")
        
        # Initialize model
        model = get_model(model_name, config)
        model = model.to(device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            model.parameters(),
            lr=model_config['learning_rate'],
            weight_decay=model_config['weight_decay']
        )
        
        if config['training']['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs']
            )
        
        # Initialize loss function and scaler
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=config['training']['mixed_precision'])
        
        # Train on each dataset and resolution
        for dataset_name, dataset_config in config['datasets'].items():
            print(f"\nTraining on {dataset_name}...")
            
            # Train on original resolution
            train_loader = get_dataloader(
                dataset_name,
                'data',
                dataset_config['batch_size'],
                dataset_config['original_size'],
                train=True,
                num_workers=dataset_config['num_workers']
            )
            
            test_loader = get_dataloader(
                dataset_name,
                'data',
                dataset_config['batch_size'],
                dataset_config['original_size'],
                train=False,
                num_workers=dataset_config['num_workers']
            )
            
            # Training loop
            best_acc = 0
            for epoch in range(config['training']['epochs']):
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, scaler, device, config
                )
                
                test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                
                if config['training']['scheduler'] == 'cosine':
                    scheduler.step()
                
                # Log metrics
                if config['logging']['wandb']['enabled']:
                    wandb.log({
                        f'{model_name}/{dataset_name}/train_loss': train_loss,
                        f'{model_name}/{dataset_name}/train_acc': train_acc,
                        f'{model_name}/{dataset_name}/test_loss': test_loss,
                        f'{model_name}/{dataset_name}/test_acc': test_acc,
                        'epoch': epoch
                    })
                
                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'test_acc': test_acc
                    }, results_dir / f'{model_name}_{dataset_name}_best.pth')
                
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Train on downsampled versions
            for size in dataset_config['downsampled_sizes']:
                print(f"\nTraining on {dataset_name} at {size}x{size}...")
                
                train_loader = get_dataloader(
                    dataset_name,
                    'data',
                    dataset_config['batch_size'],
                    size,
                    train=True,
                    num_workers=dataset_config['num_workers']
                )
                
                test_loader = get_dataloader(
                    dataset_name,
                    'data',
                    dataset_config['batch_size'],
                    size,
                    train=False,
                    num_workers=dataset_config['num_workers']
                )
                
                # Reset model and optimizer
                model = get_model(model_name, config)
                model = model.to(device)
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=model_config['learning_rate'],
                    weight_decay=model_config['weight_decay']
                )
                
                # Training loop for downsampled version
                best_acc = 0
                for epoch in range(config['training']['epochs']):
                    train_loss, train_acc = train_epoch(
                        model, train_loader, criterion, optimizer, scaler, device, config
                    )
                    
                    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
                    
                    if config['training']['scheduler'] == 'cosine':
                        scheduler.step()
                    
                    # Log metrics
                    if config['logging']['wandb']['enabled']:
                        wandb.log({
                            f'{model_name}/{dataset_name}_{size}/train_loss': train_loss,
                            f'{model_name}/{dataset_name}_{size}/train_acc': train_acc,
                            f'{model_name}/{dataset_name}_{size}/test_loss': test_loss,
                            f'{model_name}/{dataset_name}_{size}/test_acc': test_acc,
                            'epoch': epoch
                        })
                    
                    # Save best model
                    if test_acc > best_acc:
                        best_acc = test_acc
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'test_acc': test_acc
                        }, results_dir / f'{model_name}_{dataset_name}_{size}_best.pth')
                    
                    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                          f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    if config['logging']['wandb']['enabled']:
        wandb.finish()

if __name__ == '__main__':
    main() 