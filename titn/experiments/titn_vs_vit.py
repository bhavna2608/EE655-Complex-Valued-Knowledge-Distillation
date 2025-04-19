import torch
import matplotlib.pyplot as plt
from models import TITN
from models.teacher_models import get_vit_model
from data import build_dataset
from utils.helpers import load_config
from training.train import train_epoch, validate

def train_model(model, train_loader, val_loader, device='cuda'):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    val_acc = []
    for epoch in range(10):  # Reduced epochs
        train_epoch(
            model, None, train_loader, 
            optimizer, torch.nn.CrossEntropyLoss(), 
            device, epoch
        )
        _, acc1, _ = validate(
            model, val_loader, 
            torch.nn.CrossEntropyLoss(), device
        )
        val_acc.append(acc1)
    return val_acc

def run_titn_vs_vit():
    datasets = ['cifar10', 'cifar100', 'mnist']
    results = {'TITN': {}, 'ViT': {}}

    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        config = load_config(f'configs/{dataset}.yaml')
        train_set, val_set = build_dataset(config)
        
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4
        )
        val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Train TITN
        print("Training TITN...")
        titn_model = TITN(config).cuda()
        results['TITN'][dataset] = train_model(titn_model, train_loader, val_loader)
        
        # Train ViT
        print("Training ViT...")
        vit_model = get_vit_model(config).cuda()
        results['ViT'][dataset] = train_model(vit_model, train_loader, val_loader)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    for i, dataset in enumerate(datasets):
        axs[i].plot(results['TITN'][dataset], label='TITN')
        axs[i].plot(results['ViT'][dataset], label='ViT')
        axs[i].set_title(dataset.upper())
        axs[i].set_xlabel('Epochs')
        axs[i].set_ylabel('Accuracy')
        axs[i].legend()
    plt.tight_layout()
    plt.savefig('results/titn_vs_vit.png')
    plt.close()

if __name__ == '__main__':
    run_titn_vs_vit()