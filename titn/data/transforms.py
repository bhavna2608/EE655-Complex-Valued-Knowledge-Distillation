import torchvision.transforms as transforms
from .augmentations import AutoAugment

def build_transforms(config):
    """Build train and test transforms based on config"""
    mean = {
        'cifar10': [0.4914, 0.4822, 0.4465],
        'cifar100': [0.5071, 0.4867, 0.4408],
        'mnist': [0.1307]
    }[config.dataset]
    
    std = {
        'cifar10': [0.2470, 0.2435, 0.2616],
        'cifar100': [0.2675, 0.2565, 0.2761],
        'mnist': [0.3081]
    }[config.dataset]
    
    # Common test transforms
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Train transforms
    if config.dataset == 'mnist':
        # MNIST specific transforms from paper
        train_transforms = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # CIFAR transforms
        base_transforms = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        
        # Add AutoAugment if enabled
        if config.use_autoaugment:
            base_transforms.insert(0, AutoAugment())
        
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        train_transforms = transforms.Compose(base_transforms)
    
    return train_transforms, test_transforms