import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def create_optimizer(model, config):
    """Creates optimizer and scheduler as per paper specs"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=1e-6  # Minimum learning rate
    )
    
    return optimizer, scheduler