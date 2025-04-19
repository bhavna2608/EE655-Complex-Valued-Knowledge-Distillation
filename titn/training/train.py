import torch
from tqdm import tqdm
from .metrics import accuracy
from .losses import HybridDistillationLoss

def train_epoch(model, teacher_model, train_loader, optimizer, loss_fn, device, epoch):
    model.train()
    
    # Only call eval() on teacher_model if it's not None
    if teacher_model:
        teacher_model.eval()
    
    total_loss = 0
    top1_acc = 0
    top5_acc = 0

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    
    for batch_idx, (data, target) in enumerate(pbar):
        print(f"Data type: {type(data)}, Target type: {type(target)}")
        print(f"Data shape: {data.shape}, Target shape: {target.shape}")
        data, target = data.to(device), target.to(device)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        student_output = model(data)
        loss = loss_fn(data, student_output, target, teacher_model)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        cls_out = student_output[0]  # Classification output
        acc1, acc5 = accuracy(cls_out, target, topk=(1, 5))
        total_loss += loss.item()
        top1_acc += acc1.item()
        top5_acc += acc5.item()
        
        pbar.set_postfix({
            'loss': loss.item(),
            'top1': acc1.item(),
            'top5': acc5.item()
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_top1 = top1_acc / len(train_loader)
    avg_top5 = top5_acc / len(train_loader)
    
    return avg_loss, avg_top1, avg_top5

def validate(model, val_loader, loss_fn, device):
    model.eval()
    
    total_loss = 0
    top1_acc = 0
    top5_acc = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]  # Take classification output
            
            loss = loss_fn(data, output, target, None)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_loss += loss.item()
            top1_acc += acc1.item()
            top5_acc += acc5.item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'top1': acc1.item(),
                'top5': acc5.item()
            })
    
    avg_loss = total_loss / len(val_loader)
    avg_top1 = top1_acc / len(val_loader)
    avg_top5 = top5_acc / len(val_loader)
    
    return avg_loss, avg_top1, avg_top5