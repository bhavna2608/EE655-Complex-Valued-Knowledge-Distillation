import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset, ConcatDataset
from .transforms import build_transforms
from .augmentations import CutMix
import random


class CIFAR10WithCutMix(Dataset):
    def __init__(self, root, train=True, transform=None, cutmix=None):
        self.dataset = datasets.CIFAR10(root=root, train=train, download=True)
        self.transform = transform
        self.cutmix = cutmix
        self.targets = self.dataset.targets
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        if self.cutmix and torch.rand(1) < 0.5:  # 50% chance of CutMix
            rand_index = torch.randperm(len(self.dataset))[0]
            img2, label2 = self.dataset[rand_index]
            img, (label, label2, lam) = self.cutmix(img, img2, label, label2)
            if self.transform:
                img = self.transform(img)
            return img, (label, label2, lam)
        
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.dataset)

class CIFAR100WithCutMix(Dataset):
    def __init__(self, root, train=True, transform=None, cutmix=None, cutmix_prob=0.5):
        """
        Args:
            root (string): Root directory of dataset.
            train (bool, optional): If True, uses the train set. Default is True.
            transform (callable, optional): Optional transform to be applied on a sample.
            cutmix (CutMix object): CutMix augmentation to apply.
            cutmix_prob (float): Probability of applying CutMix.
        """
        self.dataset = datasets.CIFAR100(root=root, train=train, download=True)
        self.transform = transform
        self.cutmix = cutmix
        self.cutmix_prob = cutmix_prob
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        # Initialize label2 and lam to None
        label2, lam = None, None
        
        # Apply CutMix with probability cutmix_prob
        if self.cutmix and random.random() < self.cutmix_prob:
            rand_index = random.randint(0, len(self.dataset) - 1)
            img2, label2 = self.dataset[rand_index]
            img, (label, label2, lam) = self.cutmix(img, img2, label, label2)
        
        if self.transform:
            img = self.transform(img)

        # Return the proper value
        if label2 is not None:
            return img, (label, label2, lam)  # CutMix case
        else:
            return img, label  # Normal case

    def __len__(self):
        return len(self.dataset)

def build_dataset(config):
    """Build dataset based on config"""
    # Get transforms
    train_transforms, test_transforms = build_transforms(config)
    
    # Initialize CutMix if enabled
    cutmix = CutMix(alpha=config.cutmix_alpha) if config.use_cutmix else None
    
    # Dataset specific logic
    if config.dataset == 'cifar10':
        train_set = CIFAR10WithCutMix(
            root = str(config.data.root),
            train=True,
            transform=train_transforms,
            cutmix=cutmix
        )
        test_set = datasets.CIFAR10(
            root = str(config.data.root),
            train=False,
            transform=test_transforms
        )
    elif config.dataset == 'cifar100':
        train_set = CIFAR100WithCutMix(  # Similar to CIFAR10 implementation
            root = str(config.data.root),
            train=True,
            transform=train_transforms,
            cutmix=cutmix
        )
        test_set = datasets.CIFAR100(
            root = str(config.data.root),
            train=False,
            transform=test_transforms
        )
    elif config.dataset == 'mnist':
        # MNIST uses different augmentations (from paper Section 4.1)
        train_set = datasets.MNIST(
            root = str(config.data.root),
            train=True,
            transform=train_transforms,
            download=True
        )
        test_set = datasets.MNIST(
            root = str(config.data.root),
            train=False,
            transform=test_transforms
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")
    
    return train_set, test_set