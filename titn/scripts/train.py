#!/usr/bin/env python3
import argparse
import os
from utils.helpers import (
    seed_everything, 
    save_config,
    load_config,
    save_checkpoint,
    load_checkpoint
)
from utils.logger import log_metrics
from training.train import validate
import torch
from utils.helpers import seed_everything, save_config
from utils.logger import setup_logging
from data import build_dataset
from models import TITN
from models.teacher_models import get_teacher_model
from training.train import train_epoch, validate
from training.optimizer import create_optimizer
from training.losses import HybridDistillationLoss

def main():
    parser = argparse.ArgumentParser(description='TITN Training')
    parser.add_argument('--config', default='configs/default.yaml', help='config file path')
    parser.add_argument('--resume', help='checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logging
    writer = setup_logging(config)
    save_config(config, os.path.join(writer.log_dir, 'config.yaml'))

    # Build datasets
    train_set, val_set = build_dataset(config)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Initialize models
    model = TITN(config).to(device)
    teacher = get_teacher_model(config).to(device)
    loss_fn = HybridDistillationLoss(alpha=config.alpha)

    # Create optimizer
    optimizer, scheduler = create_optimizer(model, config)

    # Resume if specified
    start_epoch = 0
    best_acc = 0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(model, optimizer, args.resume, device)

    # Training loop
    for epoch in range(start_epoch, config.epochs):
        # Train epoch
        train_loss, train_top1, train_top5 = train_epoch(
            model, teacher, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_loss, val_top1, val_top5 = validate(model, val_loader, loss_fn, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        log_metrics(writer, {
            'loss': train_loss,
            'top1_acc': train_top1,
            'top5_acc': train_top5
        }, epoch, 'train')
        
        log_metrics(writer, {
            'loss': val_loss,
            'top1_acc': val_top1,
            'top5_acc': val_top5
        }, epoch, 'val')
        
        # Save checkpoint
        if val_top1 > best_acc:
            best_acc = val_top1
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(writer.log_dir, 'best_model.pth'))

    writer.close()

if __name__ == '__main__':
    main()