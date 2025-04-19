import torch
import matplotlib.pyplot as plt
from titn.models import TITN, ViT
from titn.models.c_titn import ComplexTITN
from titn.data import build_dataset
from titn.utils.helpers import load_config
from titn.training.train import train_epoch, validate

def train_complex_model(model, train_loader, val_loader, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = torch.nn.CrossEntropyLoss()
    
    val_acc = []
    for epoch in range(config["epochs"]):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            print(f"Output shape: {out.shape}")
            # Assuming real and imaginary parts are in the first and second channels
            loss = criterion(out[..., 0], y) + criterion(out[..., 1], y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out[..., 0].argmax(dim=-1)  # Use real part for prediction
                correct += (preds == y).sum().item()
                total += y.size(0)
        
        val_acc.append(100 * correct / total)
        print(f"Epoch {epoch+1}/{config["epochs"]} | Val Acc: {val_acc[-1]:.2f}%")
    
    return val_acc

def run_complex_comparison():
    config = load_config('titn/configs/cifar100.yaml')
    config["epochs"] = 100  # Reduced for demo
    
    # Prepare data
    train_set, val_set = build_dataset(config)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config["batch_size"])
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    models = {
        'ViT': ViT(config).to(device),
        'TITN': TITN(config).to(device),
        'c-TITN': ComplexTITN(config).to(device)
    }
    
    # Train and compare
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        if 'c-' in name:
            results[name] = train_complex_model(model, train_loader, val_loader, config, device)
        else:
            results[name] = train_epoch(model, train_loader, val_loader, config, device)
    
    # Plot results
    plt.figure(figsize=(10,6))
    for name, acc in results.items():
        plt.plot(acc, label=name)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Complex vs Real Models Comparison')
    plt.legend()
    plt.savefig('results/complex_comparison.png')
    plt.close()

if __name__ == '__main__':
    run_complex_comparison()
