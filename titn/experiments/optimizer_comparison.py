import torch
import matplotlib.pyplot as plt
from models.teacher_models import get_teacher_model
from data import build_dataset
from utils.helpers import load_config
from training.train import train_epoch, validate

def run_optimizer_comparison():
    # Load config and data
    config = load_config('configs/cifar100.yaml')
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

    optimizers = {
        'SGD': torch.optim.SGD,
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop
    }
    
    results = {}
    for name, opt_class in optimizers.items():
        print(f"\nTraining with {name}...")
        config.teacher.name = 'vgg16_bn'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_teacher_model(config).to(device)
        optimizer = opt_class(model.parameters(), lr=0.001)
        
        val_acc = []
        for epoch in range(10):  # Reduced epochs
            # Train
            train_epoch(
                model, None, train_loader, 
                optimizer, torch.nn.CrossEntropyLoss(), 
                'cuda', epoch
            )
            # Validate
            _, acc1, _ = validate(
                model, val_loader, 
                torch.nn.CrossEntropyLoss(), 'cuda'
            )
            val_acc.append(acc1)
        
        results[name] = val_acc

    # Plot results
    plt.figure(figsize=(10,5))
    for name, acc in results.items():
        plt.plot(acc, label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Optimizer Comparison (VGG16-bn)')
    plt.legend()
    plt.savefig('results/optimizer_comparison.png')
    plt.close()

if __name__ == '__main__':
    run_optimizer_comparison()