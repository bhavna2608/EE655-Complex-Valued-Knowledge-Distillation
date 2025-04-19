import os
import random
import numpy as np
import torch
import yaml
from addict import Dict

def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """Save training checkpoint"""
    torch.save(state, filename)

def load_config(config_path):
    """Load and merge YAML config files"""
    with open(config_path) as f:
        config = Dict(yaml.safe_load(f))
    
    # If there's a base config specified, merge it
    if 'base_config' in config:
        base_config = load_config(config.base_config)
        base_config.update(config)
        return base_config
    
    return config

def save_config(config, path):
    """Save config to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f)