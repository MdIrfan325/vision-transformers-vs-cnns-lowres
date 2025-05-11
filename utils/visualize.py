import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import torch
from PIL import Image
import torchvision.transforms as T

def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all results from the results directory."""
    results = {}
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data = json.load(f)
                    key = os.path.splitext(file)[0]
                    results[key] = data
    return results

def plot_training_curves(results: Dict[str, Any], output_dir: Path):
    """Plot training curves for each model and dataset."""
    for key, data in results.items():
        if 'training_history' in data:
            plt.figure(figsize=(10, 6))
            
            # Plot training and validation accuracy
            plt.subplot(1, 2, 1)
            plt.plot(data['training_history']['train_acc'], label='Train')
            plt.plot(data['training_history']['val_acc'], label='Validation')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            
            # Plot training and validation loss
            plt.subplot(1, 2, 2)
            plt.plot(data['training_history']['train_loss'], label='Train')
            plt.plot(data['training_history']['val_loss'], label='Validation')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_dir / f"training_curves_{key}.png")
            plt.close()

def plot_confusion_matrices(results: Dict[str, Any], output_dir: Path):
    """Plot confusion matrices for each model and dataset."""
    for key, data in results.items():
        if 'confusion_matrix' in data:
            cm = np.array(data['confusion_matrix'])
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {key}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(output_dir / f"confusion_matrix_{key}.png")
            plt.close()

def plot_noise_robustness(results: Dict[str, Any], output_dir: Path):
    """Plot noise robustness curves for each model and dataset."""
    noise_types = ['gaussian', 'salt_pepper']
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    for dataset in ['cifar10', 'fashion_mnist', 'svhn']:
        for resolution in [32, 16, 8] if dataset != 'fashion_mnist' else [28, 14, 7]:
            plt.figure(figsize=(12, 6))
            
            for noise_type in noise_types:
                plt.subplot(1, 2, 1 if noise_type == 'gaussian' else 2)
                for model in ['lenet5', 'resnet18', 'efficientnetv2_s', 'mobilenetv2', 'convmixer', 'repvgg',
                            'vit', 'deit', 'mobilevit', 'swin', 'cvt']:
                    key = f"{model}_{dataset}_{resolution}"
                    if key in results and 'noise_robustness' in results[key]:
                        accuracies = [results[key]['noise_robustness'].get(f"{noise_type}_{level}", 0.0)
                                    for level in noise_levels]
                        plt.plot(noise_levels, accuracies, marker='o', label=model.replace('_', ' ').title())
                
                plt.title(f'{noise_type.replace("_", " ").title()} Noise')
                plt.xlabel('Noise Level')
                plt.ylabel('Accuracy (%)')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / f"noise_robustness_{dataset}_{resolution}.png", bbox_inches='tight')
            plt.close()

def plot_model_comparison(results: Dict[str, Any], output_dir: Path):
    """Plot model comparison bar charts for each metric."""
    metrics = ['accuracy', 'model_size', 'flops', 'inference_time']
    datasets = ['cifar10', 'fashion_mnist', 'svhn']
    
    for dataset in datasets:
        for resolution in [32, 16, 8] if dataset != 'fashion_mnist' else [28, 14, 7]:
            for metric in metrics:
                plt.figure(figsize=(15, 6))
                
                models = []
                values = []
                for model in ['lenet5', 'resnet18', 'efficientnetv2_s', 'mobilenetv2', 'convmixer', 'repvgg',
                            'vit', 'deit', 'mobilevit', 'swin', 'cvt']:
                    key = f"{model}_{dataset}_{resolution}"
                    if key in results and metric in results[key]:
                        models.append(model.replace('_', ' ').title())
                        values.append(results[key][metric])
                
                plt.bar(models, values)
                plt.title(f'{metric.replace("_", " ").title()} Comparison - {dataset.upper()} {resolution}x{resolution}')
                plt.xticks(rotation=45, ha='right')
                plt.ylabel(metric.replace('_', ' ').title())
                plt.tight_layout()
                plt.savefig(output_dir / f"model_comparison_{metric}_{dataset}_{resolution}.png")
                plt.close()

def visualize_attention_maps(model, image, output_dir: Path, model_name: str, dataset: str, resolution: int):
    """Visualize attention maps for transformer models."""
    if hasattr(model, 'get_attention_maps'):
        attention_maps = model.get_attention_maps(image)
        if attention_maps is not None:
            for i, attn_map in enumerate(attention_maps):
                plt.figure(figsize=(8, 8))
                plt.imshow(attn_map[0].detach().cpu().numpy(), cmap='viridis')
                plt.title(f'Attention Map {i+1}')
                plt.colorbar()
                plt.savefig(output_dir / f"attention_map_{model_name}_{dataset}_{resolution}_{i+1}.png")
                plt.close()

def visualize_feature_maps(model, image, output_dir: Path, model_name: str, dataset: str, resolution: int):
    """Visualize feature maps for all models."""
    if hasattr(model, 'get_feature_maps'):
        feature_maps = model.get_feature_maps(image)
        if feature_maps is not None:
            for i, feat_map in enumerate(feature_maps):
                # Select first channel for visualization
                feat = feat_map[0, 0].detach().cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.imshow(feat, cmap='viridis')
                plt.title(f'Feature Map {i+1}')
                plt.colorbar()
                plt.savefig(output_dir / f"feature_map_{model_name}_{dataset}_{resolution}_{i+1}.png")
                plt.close()

def main():
    # Load results
    results_dir = "results"
    results = load_results(results_dir)
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations
    plot_training_curves(results, output_dir)
    plot_confusion_matrices(results, output_dir)
    plot_noise_robustness(results, output_dir)
    plot_model_comparison(results, output_dir)
    
    # Note: Attention and feature map visualizations require model instances and input images
    # These should be called during model evaluation

if __name__ == "__main__":
    main() 