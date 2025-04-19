import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        if target.dim() == 2:  # For CutMix labels
            target = target.argmax(1)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_metrics(output, target):
    """Returns dictionary of metrics"""
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    return {
        'top1': acc1.item(),
        'top5': acc5.item()
    }