import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

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

def format_metric(value: float, metric: str) -> str:
    """Format a metric value for LaTeX table."""
    if metric == 'accuracy':
        return f"{value:.2f}\\%"
    elif metric == 'model_size':
        return f"{value:.2f}MB"
    elif metric == 'flops':
        return f"{value:.2f}M"
    elif metric == 'inference_time':
        return f"{value:.2f}ms"
    elif metric == 'noise_robustness':
        return f"{value:.2f}\\%"
    else:
        return f"{value:.2f}"

def generate_model_comparison_table(results: Dict[str, Any], dataset: str, resolution: int) -> str:
    """Generate LaTeX table comparing models on a specific dataset and resolution."""
    models = ['lenet5', 'resnet18', 'efficientnetv2_s', 'mobilenetv2', 'convmixer', 'repvgg',
              'vit', 'deit', 'mobilevit', 'swin', 'cvt']
    metrics = ['accuracy', 'model_size', 'flops', 'inference_time', 'noise_robustness']
    
    # Table header
    table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
    table += "\\hline\n"
    table += "Model & " + " & ".join(metric.replace('_', ' ').title() for metric in metrics) + " \\\\\n"
    table += "\\hline\n"
    
    # Table content
    for model in models:
        key = f"{model}_{dataset}_{resolution}"
        if key in results:
            row = [model.replace('_', ' ').title()]
            for metric in metrics:
                value = results[key].get(metric, 0.0)
                row.append(format_metric(value, metric))
            table += " & ".join(row) + " \\\\\n"
    
    # Table footer
    table += "\\hline\n\\end{tabular}\n"
    table += f"\\caption{{Model comparison on {dataset.upper()} at {resolution}x{resolution} resolution}}\n"
    table += "\\label{tab:" + f"{dataset}_{resolution}" + "}\n"
    table += "\\end{table}\n"
    
    return table

def generate_resolution_comparison_table(results: Dict[str, Any], model: str, dataset: str) -> str:
    """Generate LaTeX table comparing model performance across resolutions."""
    resolutions = [32, 16, 8] if dataset != 'fashion_mnist' else [28, 14, 7]
    metrics = ['accuracy', 'model_size', 'flops', 'inference_time', 'noise_robustness']
    
    # Table header
    table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{l" + "c" * len(resolutions) + "}\n"
    table += "\\hline\n"
    table += "Metric & " + " & ".join(f"{r}x{r}" for r in resolutions) + " \\\\\n"
    table += "\\hline\n"
    
    # Table content
    for metric in metrics:
        row = [metric.replace('_', ' ').title()]
        for resolution in resolutions:
            key = f"{model}_{dataset}_{resolution}"
            if key in results:
                value = results[key].get(metric, 0.0)
                row.append(format_metric(value, metric))
            else:
                row.append("-")
        table += " & ".join(row) + " \\\\\n"
    
    # Table footer
    table += "\\hline\n\\end{tabular}\n"
    table += f"\\caption{{{model.replace('_', ' ').title()} performance across resolutions on {dataset.upper()}}}\n"
    table += "\\label{tab:" + f"{model}_{dataset}" + "}\n"
    table += "\\end{table}\n"
    
    return table

def generate_noise_robustness_table(results: Dict[str, Any], dataset: str, resolution: int) -> str:
    """Generate LaTeX table comparing noise robustness across models."""
    models = ['lenet5', 'resnet18', 'efficientnetv2_s', 'mobilenetv2', 'convmixer', 'repvgg',
              'vit', 'deit', 'mobilevit', 'swin', 'cvt']
    noise_types = ['gaussian', 'salt_pepper']
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Table header
    table = "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{l" + "c" * (len(noise_types) * len(noise_levels)) + "}\n"
    table += "\\hline\n"
    table += "Model & " + " & ".join([f"{t} {l}" for t in noise_types for l in noise_levels]) + " \\\\\n"
    table += "\\hline\n"
    
    # Table content
    for model in models:
        key = f"{model}_{dataset}_{resolution}"
        if key in results and 'noise_robustness' in results[key]:
            row = [model.replace('_', ' ').title()]
            for noise_type in noise_types:
                for level in noise_levels:
                    value = results[key]['noise_robustness'].get(f"{noise_type}_{level}", 0.0)
                    row.append(format_metric(value, 'noise_robustness'))
            table += " & ".join(row) + " \\\\\n"
    
    # Table footer
    table += "\\hline\n\\end{tabular}\n"
    table += f"\\caption{{Noise robustness comparison on {dataset.upper()} at {resolution}x{resolution} resolution}}\n"
    table += "\\label{tab:" + f"noise_{dataset}_{resolution}" + "}\n"
    table += "\\end{table}\n"
    
    return table

def main():
    # Load results
    results_dir = "results"
    results = load_results(results_dir)
    
    # Create output directory
    output_dir = Path("tables")
    output_dir.mkdir(exist_ok=True)
    
    # Generate tables for each dataset and resolution
    datasets = ['cifar10', 'fashion_mnist', 'svhn']
    for dataset in datasets:
        resolutions = [32, 16, 8] if dataset != 'fashion_mnist' else [28, 14, 7]
        
        # Model comparison tables
        for resolution in resolutions:
            table = generate_model_comparison_table(results, dataset, resolution)
            with open(output_dir / f"model_comparison_{dataset}_{resolution}.tex", 'w') as f:
                f.write(table)
        
        # Resolution comparison tables
        models = ['lenet5', 'resnet18', 'efficientnetv2_s', 'mobilenetv2', 'convmixer', 'repvgg',
                 'vit', 'deit', 'mobilevit', 'swin', 'cvt']
        for model in models:
            table = generate_resolution_comparison_table(results, model, dataset)
            with open(output_dir / f"resolution_comparison_{model}_{dataset}.tex", 'w') as f:
                f.write(table)
        
        # Noise robustness tables
        for resolution in resolutions:
            table = generate_noise_robustness_table(results, dataset, resolution)
            with open(output_dir / f"noise_robustness_{dataset}_{resolution}.tex", 'w') as f:
                f.write(table)

if __name__ == "__main__":
    main() 