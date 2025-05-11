import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
from sklearn.metrics import confusion_matrix, classification_report
import cv2
import json
import importlib

from data.datasets import get_dataloader, add_noise, load_config
from models.cnn.lenet5 import LeNet5
from models.cnn.resnet import ModifiedResNet18
from models.transformer.vit import VisionTransformer
from utils.visualize import visualize_attention_maps, visualize_feature_maps

def get_model(model_name: str, model_config: dict, num_classes: int) -> nn.Module:
    """Dynamically import and instantiate model."""
    model_type = model_config['type']
    model_module = importlib.import_module(f'models.{model_type}.{model_name}')
    model_class = getattr(model_module, model_name.replace('-', '').title())
    return model_class(num_classes=num_classes, **model_config['params'])

def evaluate_model(model: nn.Module,
                  test_loader: torch.utils.data.DataLoader,
                  criterion: nn.Module,
                  device: torch.device) -> dict:
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
    conf_matrix = confusion_matrix(all_targets, all_preds)
    class_report = classification_report(all_targets, all_preds, output_dict=True)
    
    return {
        'loss': total_loss / len(test_loader),
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report
    }

def evaluate_noise_robustness(model, test_loader, device, config):
    """Evaluate model's robustness to different types of noise."""
    results = {}
    
    # Gaussian noise
    for std in config['evaluation']['noise_test']['gaussian_noise_std']:
        noisy_loader = torch.utils.data.DataLoader(
            test_loader.dataset,
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=test_loader.num_workers
        )
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in noisy_loader:
                inputs = add_noise(inputs, noise_type='gaussian', noise_param=std)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        results[f'gaussian_noise_{std}'] = 100. * correct / total
    
    # Salt and pepper noise
    for prob in config['evaluation']['noise_test']['salt_pepper_prob']:
        noisy_loader = torch.utils.data.DataLoader(
            test_loader.dataset,
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=test_loader.num_workers
        )
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in noisy_loader:
                inputs = add_noise(inputs, noise_type='salt_pepper', noise_param=prob)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        results[f'salt_pepper_noise_{prob}'] = 100. * correct / total
    
    return results

def visualize_attention(model, test_loader, device, save_dir):
    """Visualize attention maps for transformer models."""
    if not hasattr(model, 'get_attention_maps'):
        return
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 5:  # Visualize only first 5 examples
                break
            
            inputs = inputs.to(device)
            attn_maps = model.get_attention_maps(inputs)
            
            # Save attention maps
            for layer_idx, attn in enumerate(attn_maps):
                # Get attention weights for class token
                attn = attn[0, 0, 1:, 1:]  # Remove class token attention
                
                # Reshape to image size
                img_size = int(np.sqrt(attn.size(0)))
                attn = attn.reshape(img_size, img_size)
                
                # Normalize and convert to heatmap
                attn = (attn - attn.min()) / (attn.max() - attn.min())
                attn = (attn * 255).astype(np.uint8)
                attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
                
                # Save attention map
                save_path = save_dir / f'attention_layer{layer_idx}_sample{i}.png'
                cv2.imwrite(str(save_path), attn)

def visualize_gradcam(model, test_loader, device, save_dir):
    """Visualize Grad-CAM for CNN models."""
    if not hasattr(model, 'get_feature_maps'):
        return
    
    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if i >= 5:  # Visualize only first 5 examples
                break
            
            inputs = inputs.to(device)
            features = model.get_feature_maps(inputs)
            
            # Save feature maps
            for layer_idx, feat in enumerate(features):
                # Get mean activation
                feat = feat[0].mean(0).cpu().numpy()
                
                # Normalize and convert to heatmap
                feat = (feat - feat.min()) / (feat.max() - feat.min())
                feat = (feat * 255).astype(np.uint8)
                feat = cv2.applyColorMap(feat, cv2.COLORMAP_JET)
                
                # Save feature map
                save_path = save_dir / f'gradcam_layer{layer_idx}_sample{i}.png'
                cv2.imwrite(str(save_path), feat)

def plot_confusion_matrix(conf_matrix: np.ndarray,
                         class_names: list,
                         output_path: Path,
                         title: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Load configuration
    config = load_config('configs/experiment_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create results directory
    results_dir = Path(config['logging']['save_dir']) / 'evaluation'
    results_dir.mkdir(exist_ok=True)
    
    # Initialize results dictionary
    results = {}
    
    # Evaluation loop
    for dataset_name, dataset_config in config['datasets'].items():
        if not dataset_config['enabled']:
            continue
            
        results[dataset_name] = {}
        
        for target_size in dataset_config['target_sizes']:
            results[dataset_name][str(target_size)] = {}
            
            # Get test dataloader
            test_loader, num_classes = get_dataloader(
                dataset_name=dataset_name,
                root_dir=dataset_config['root_dir'],
                batch_size=config['training']['batch_size'],
                is_train=False,
                target_size=target_size,
                config=config
            )
            
            # Get class names
            if dataset_name == 'cifar10':
                class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                             'dog', 'frog', 'horse', 'ship', 'truck']
            elif dataset_name == 'fashion_mnist':
                class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                             'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            elif dataset_name == 'svhn':
                class_names = [str(i) for i in range(10)]
            
            # Evaluate each model
            for model_name, model_config in config['models'].items():
                if not model_config['enabled']:
                    continue
                
                print(f"\nEvaluating {model_name} on {dataset_name} at {target_size}x{target_size}")
                
                # Load model
                model = get_model(model_name, model_config, num_classes)
                model = model.to(device)
                
                # Load best weights
                checkpoint_path = Path(config['logging']['save_dir']) / f'{model_name}_{dataset_name}_{target_size}.pth'
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    print(f"No checkpoint found for {model_name} on {dataset_name} at {target_size}x{target_size}")
                    continue
                
                # Evaluate
                criterion = nn.CrossEntropyLoss()
                metrics = evaluate_model(model, test_loader, criterion, device)
                
                # Save results
                results[dataset_name][str(target_size)][model_name] = {
                    'accuracy': float(metrics['accuracy']),
                    'loss': float(metrics['loss']),
                    'classification_report': metrics['classification_report']
                }
                
                # Plot confusion matrix
                conf_matrix_path = results_dir / f'{model_name}_{dataset_name}_{target_size}_confusion_matrix.png'
                plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    class_names,
                    conf_matrix_path,
                    f'{model_name} on {dataset_name} at {target_size}x{target_size}'
                )
                
                # Visualize attention maps for transformer models
                if model_config['type'] == 'transformer':
                    # Get a sample batch
                    sample_inputs, _ = next(iter(test_loader))
                    sample_input = sample_inputs[0].unsqueeze(0).to(device)
                    
                    # Save attention maps
                    attention_path = results_dir / f'{model_name}_{dataset_name}_{target_size}_attention.png'
                    visualize_attention_maps(
                        model, sample_input, attention_path,
                        model_name, dataset_name, target_size
                    )
                
                # Visualize feature maps
                feature_path = results_dir / f'{model_name}_{dataset_name}_{target_size}_features.png'
                visualize_feature_maps(
                    model, sample_input, feature_path,
                    model_name, dataset_name, target_size
                )
    
    # Save all results
    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main() 