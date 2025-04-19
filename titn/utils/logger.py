import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def setup_logging(config):
    """Setup logging and tensorboard writer"""
    # Create log directory
    log_dir = os.path.join(config.log_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup file logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    
    return writer

def log_metrics(writer, metrics, epoch, phase='train'):
    """Log metrics to tensorboard and console"""
    # Log to tensorboard
    for metric_name, metric_value in metrics.items():
        writer.add_scalar(f'{phase}/{metric_name}', metric_value, epoch)
    
    # Log to console
    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    logging.info(f'Epoch {epoch} {phase} - {metrics_str}')