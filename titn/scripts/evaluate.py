#!/usr/bin/env python3
import argparse
import torch
from utils.helpers import load_checkpoint
from data import build_dataset
from models import TITN
from training.metrics import accuracy
from training.losses import HybridDistillationLoss
from utils.helpers import load_config
from training.train import validate
from models import TITN 

def evaluate(config, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Build dataset
    _, test_set = build_dataset(config)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = TITN(config).to(device)
    loss_fn = HybridDistillationLoss(alpha=config.alpha)
    
    # Load checkpoint
    load_checkpoint(model, None, checkpoint_path, device)
    
    # Evaluate
    test_loss, test_top1, test_top5 = validate(model, test_loader, loss_fn, device)
    
    print(f"Test Results - Loss: {test_loss:.4f} | Top1 Acc: {test_top1:.2f}% | Top5 Acc: {test_top5:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TITN Evaluation')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='model checkpoint path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    evaluate(config, args.checkpoint)