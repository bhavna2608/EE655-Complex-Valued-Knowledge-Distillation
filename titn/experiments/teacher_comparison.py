import torch
import matplotlib.pyplot as plt
from types import SimpleNamespace
from titn.models.teacher_models import get_teacher_model
from titn.data import build_dataset
from titn.training.train import train_epoch, validate
from titn.utils.helpers import load_config

def run_teacher_comparison():
    # Load config and data
    config = load_config('titn/configs/cifar100.yaml')
    train_set, val_set = build_dataset(config)

    # Ensure config.teacher has required fields
    if "teacher" not in config:
        config.teacher = {}
    config.teacher.setdefault("weights", None)
    config.teacher.setdefault("freeze", False)

    # Data loaders
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

    # List of teacher models to evaluate
    teachers = ['densenet121', 'mobilenet_v3', 'resnet50', 
                'vgg16', 'vgg16_bn', 'vgg19']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    for teacher in teachers:
        print(f"\nTraining with {teacher}...")
        config.teacher["name"] = teacher

        model = get_teacher_model(config).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        train_acc, val_acc = [], []
        for epoch in range(10):  # Reduced epochs for testing
            _, acc1, _ = train_epoch(
                model, None, train_loader, optimizer,
                torch.nn.CrossEntropyLoss(), device, epoch
            )
            _, vacc1, _ = validate(
                model, val_loader,
                torch.nn.CrossEntropyLoss(), device
            )
            train_acc.append(acc1)
            val_acc.append(vacc1)

        results[teacher] = {'train': train_acc, 'val': val_acc}

    # Plot results
    plt.figure(figsize=(12, 6))
    for model_name, metrics in results.items():
        plt.plot(metrics['val'], label=model_name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Teacher Models on CIFAR-100')
    plt.legend()
    plt.savefig('results/teacher_comparison.png')
    plt.close()

if __name__ == '__main__':
    run_teacher_comparison()