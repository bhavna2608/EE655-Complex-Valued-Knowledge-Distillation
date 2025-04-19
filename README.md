# EE655-Complex-Valued-Knowledge-Distillation

An implementation of "A Transformer-in-Transformer Network Utilizing Knowledge Distillation for Image Recognition" with extensions for complex-valued networks.

## Introduction

This repository contains:

1. **TITN**: Original Transformer-in-Transformer model with knowledge distillation
2. **c-TITN**: Complex-valued extension of TITN
3. **Comparison Framework**: Scripts to compare:
   - Different teacher models (VGG, ResNet, etc.)
   - Different optimizers (SGD, Adam, RMSprop)
   - TITN vs ViT vs c-TITN across datasets

## Installation

```bash
# Create and activate environment
python -m venv titn_env
source titn_env/bin/activate  # Linux/Mac
titn_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Finding the Best Teacher

```bash
python experiments/teacher_comparison.py
```

### Optimizer Comparison

```bash
python experiments/optimizer_comparison.py
```

### TITN vs ViT

```bash
python experiments/titn_vs_vit.py
```

### Complex TITN Comparison

```bash
python experiments/complex_comparison.py
```

## Configuration
Edit YAML files in configs/ to:

1. Change model architectures

2. Adjust training parameters

3. Select datasets

## Repository Structure

```bash
TITN-Knowledge-Distillation/
│
├── configs/
│   ├── default.yaml
│   ├── cifar10.yaml
│   ├── cifar100.yaml
│   └── mnist.yaml
│
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   ├── augmentations.py
│   └── transforms.py
│
├── models/
│   ├── __init__.py
│   ├── titn.py
│   ├── vit.py
│   ├── c_titn.py
│   ├── inner_transformer.py
│   ├── outer_transformer.py
│   ├── teacher_models.py
│   ├── layers.py
│   └── complex_layers.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizer.py
│   └── distillation.py
│
├── experiments/
│   ├── teacher_comparison.py
│   ├── optimizer_comparison.py
│   ├── titn_vs_vit.py
│   └── complex_comparison.py
│
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   ├── helpers.py
│   └── visualize.py
│
├── requirements.txt
└── README.md
```

## Results

Our reproduction and extension of the Transformer-in-Transformer (TITN) knowledge distillation framework yielded several key findings:

### Teacher Model Comparison (CIFAR-100):

- VGG variants outperformed other architectures, with VGG16-bn achieving 80.12% test accuracy (vs. paper's 79.80%)

- Lightweight models showed greater variance: Squeezenet reached only 40.13% (vs. paper's 42.33%)

- Performance ranking remained consistent: VGG16-bn > VGG19 > ResNet50 > MobileNetV3 > Squeezenet

### Optimizer Comparison (VGG16-bn on CIFAR-100)

- SGD performed best (80.13% vs paper's 79.80%)

- Adam showed slight degradation (77.83% vs 78.53%)

- Relative performance maintained: SGD > Adam > RMSProp

### Complex TITN (c-TITN) Analysis

- Consistently outperformed standard TITN across all datasets

- Slower initial convergence but superior final accuracy

- Achieved 83%+ accuracy on CIFAR-10 and 99.5%+ on MNIST

- Demonstrated particular strength on CIFAR-100, suggesting complex representations help with more challenging classification tasks

These results confirm the robustness of the original TITN architecture while demonstrating that the complex-valued extension can provide consistent (though modest) improvements. The reproduction validates the paper's core claims about knowledge distillation effectiveness and transformer-in-transformer architectures.
