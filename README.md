# Vision Transformers vs CNNs on Low-Resolution Datasets

This project compares the performance of various Vision Transformer and CNN architectures on low-resolution image classification tasks. The study includes multiple datasets, resolutions, and evaluation metrics to provide a comprehensive analysis.

## Features

- **Multiple Architectures**:
  - CNN Models:
    - LeNet-5
    - Modified ResNet-18
    - EfficientNetV2-S
    - MobileNetV2
    - ConvMixer
    - RepVGG-A0
  - Transformer Models:
    - Vision Transformer (ViT)
    - DeiT-Tiny
    - MobileViT
    - Tiny Swin Transformer
    - CvT (Convolutional vision Transformer)

- **Datasets**:
  - CIFAR-10 (32x32, 16x16, 8x8)
  - Fashion-MNIST (28x28, 14x14, 7x7)
  - SVHN (32x32, 16x16, 8x8)

- **Evaluation Metrics**:
  - Classification accuracy
  - Model size
  - FLOPs
  - Inference time
  - Noise robustness
  - Confusion matrices

- **Visualizations**:
  - Training curves
  - Attention maps
  - Feature maps
  - Confusion matrices
  - Model comparison charts
  - Noise robustness plots

- **LaTeX Tables**:
  - Model comparison tables
  - Resolution comparison tables
  - Noise robustness tables

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vision-transformers-vs-cnns-lowres.git
cd vision-transformers-vs-cnns-lowres
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the experiment:
   - Edit `configs/experiment_config.yaml` to specify:
     - Datasets and resolutions
     - Models to evaluate
     - Training parameters
     - Evaluation metrics
     - Visualization settings

2. Run the experiments:
```bash
python train.py
```

3. Generate visualizations:
```bash
python utils/visualize.py
```

4. Generate LaTeX tables:
```bash
python utils/generate_tables.py
```

## Project Structure

```
.
├── configs/
│   └── experiment_config.yaml
├── data/
│   └── datasets.py
├── models/
│   ├── base.py
│   ├── cnn/
│   │   ├── lenet5.py
│   │   ├── resnet.py
│   │   ├── efficientnet.py
│   │   ├── mobilenetv2.py
│   │   ├── convmixer.py
│   │   └── repvgg.py
│   └── transformer/
│       ├── vit.py
│       ├── deit.py
│       ├── mobilevit.py
│       ├── swin.py
│       └── cvt.py
├── utils/
│   ├── visualize.py
│   └── generate_tables.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

## Results

The results of the experiments will be saved in the following directories:
- `results/`: JSON files containing experiment results
- `visualizations/`: Generated plots and visualizations
- `tables/`: LaTeX tables for the results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The implementation of models is based on their respective papers and official repositories
- Special thanks to the PyTorch community for their excellent documentation and examples

## 📋 Project Overview

This research investigates the performance of Vision Transformers (ViTs) and Convolutional Neural Networks (CNNs) on low-resolution image datasets, with a focus on:
- Performance comparison across different input resolutions (8×8, 16×16, 32×32)
- Model efficiency metrics (size, speed, FLOPs)
- Training dynamics and convergence
- Robustness to noise and data corruption

## 🗂️ Project Structure

```
├── configs/                 # Configuration files
├── data/                    # Dataset loading and preprocessing
├── models/                  # Model architectures
│   ├── cnn/                # CNN implementations
│   └── transformer/        # Transformer implementations
├── utils/                  # Utility functions
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── visualize.py           # Visualization utilities
└── requirements.txt       # Project dependencies
```

## 🚀 Getting Started

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download datasets:
```bash
python data/download_datasets.py
```

3. Train models:
```bash
python train.py --config configs/experiment_config.yaml
```

4. Evaluate models:
```bash
python evaluate.py --config configs/experiment_config.yaml
```

## 📊 Datasets

- CIFAR-10 (32×32, downsampled to 16×16, 8×8)
- Fashion-MNIST (28×28, downsampled to 14×14, 8×8)
- SVHN (32×32, downsampled to 16×16, 8×8)

## 🧠 Models

### CNN Models
- LeNet-5
- ResNet-18 (modified)
- EfficientNetV2-S
- MobileNetV2
- ConvMixer
- RepVGG-A0

### Transformer Models
- ViT-Tiny
- DeiT-Tiny
- MobileViT (Tiny & XXS)
- Tiny Swin Transformer
- CvT

## 📈 Evaluation Metrics

- Classification Accuracy
- Model Size
- FLOPs
- Training Time
- Inference Time
- Convergence Analysis
- Robustness to Noise
- Visualization (CAM/Attention Maps)

## 📝 Citation

If you use this code in your research, please cite:
```bibtex
@article{vision-transformers-vs-cnns-lowres,
  title={An Empirical Comparison of Vision Transformers and Convolutional Neural Networks on Low-Resolution Datasets for Efficient Visual Recognition},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
``` 