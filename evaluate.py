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
from sklearn.metrics import confusion_matrix
import cv2

from data.datasets import get_dataloader, add_noise
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

def evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_inference_time = np.mean(inference_times)
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    return {
        'accuracy': accuracy,
        'inference_time': avg_inference_time,
        'confusion_matrix': conf_matrix
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

def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Load configuration
    config = load_config('configs/experiment_config.yaml')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create results directory
    results_dir = Path(config['logging']['save_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Initialize results DataFrame
    results = []
    
    # Evaluate each model
    for model_name, model_config in config['models'].items():
        if not model_config['enabled']:
            continue
        
        print(f"\nEvaluating {model_name}...")
        
        # Load model
        model = get_model(model_name, config)
        model = model.to(device)
        
        # Evaluate on each dataset and resolution
        for dataset_name, dataset_config in config['datasets'].items():
            print(f"\nEvaluating on {dataset_name}...")
            
            # Evaluate on original resolution
            test_loader = get_dataloader(
                dataset_name,
                'data',
                dataset_config['batch_size'],
                dataset_config['original_size'],
                train=False,
                num_workers=dataset_config['num_workers']
            )
            
            # Load best model
            checkpoint = torch.load(results_dir / f'{model_name}_{dataset_name}_best.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Evaluate model
            metrics = evaluate_model(model, test_loader, device)
            
            # Evaluate noise robustness
            noise_metrics = evaluate_noise_robustness(model, test_loader, device, config)
            
            # Get model size and FLOPs
            model_size = model.get_model_size()
            flops, params = model.get_flops()
            
            # Save results
            results.append({
                'model': model_name,
                'dataset': dataset_name,
                'resolution': dataset_config['original_size'],
                'accuracy': metrics['accuracy'],
                'inference_time': metrics['inference_time'],
                'model_size': model_size,
                'flops': flops,
                'params': params,
                **noise_metrics
            })
            
            # Plot confusion matrix
            plot_confusion_matrix(
                metrics['confusion_matrix'],
                range(10),  # Assuming 10 classes
                results_dir / f'{model_name}_{dataset_name}_confusion_matrix.png'
            )
            
            # Visualize attention maps or Grad-CAM
            if hasattr(model, 'get_attention_maps'):
                visualize_attention(model, test_loader, device, results_dir)
            if hasattr(model, 'get_feature_maps'):
                visualize_gradcam(model, test_loader, device, results_dir)
            
            # Evaluate on downsampled versions
            for size in dataset_config['downsampled_sizes']:
                print(f"\nEvaluating on {dataset_name} at {size}x{size}...")
                
                test_loader = get_dataloader(
                    dataset_name,
                    'data',
                    dataset_config['batch_size'],
                    size,
                    train=False,
                    num_workers=dataset_config['num_workers']
                )
                
                # Load best model for this resolution
                checkpoint = torch.load(results_dir / f'{model_name}_{dataset_name}_{size}_best.pth')
                model.load_state_dict(checkpoint['model_state_dict'])
                
                # Evaluate model
                metrics = evaluate_model(model, test_loader, device)
                
                # Evaluate noise robustness
                noise_metrics = evaluate_noise_robustness(model, test_loader, device, config)
                
                # Save results
                results.append({
                    'model': model_name,
                    'dataset': dataset_name,
                    'resolution': size,
                    'accuracy': metrics['accuracy'],
                    'inference_time': metrics['inference_time'],
                    'model_size': model_size,
                    'flops': flops,
                    'params': params,
                    **noise_metrics
                })
                
                # Plot confusion matrix
                plot_confusion_matrix(
                    metrics['confusion_matrix'],
                    range(10),  # Assuming 10 classes
                    results_dir / f'{model_name}_{dataset_name}_{size}_confusion_matrix.png'
                )
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / 'results.csv', index=False)
    
    # Generate summary plots
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='model', y='accuracy', hue='resolution')
    plt.title('Model Accuracy by Resolution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_by_resolution.png')
    plt.close()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=results_df, x='inference_time', y='accuracy', hue='model', size='resolution')
    plt.title('Accuracy vs. Inference Time')
    plt.tight_layout()
    plt.savefig(results_dir / 'accuracy_vs_inference_time.png')
    plt.close()

if __name__ == '__main__':
    main() 