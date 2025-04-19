import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridDistillationLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, x, student_output, labels, teacher_model=None):
        # Student outputs
        cls_out, dist_out = student_output
        
        # Teacher predictions
        with torch.no_grad():
            teacher_out = teacher_model(x)
            teacher_labels = teacher_out.argmax(dim=1)
        
        # CutMix loss (if labels is a tuple)
        if isinstance(labels, tuple):
            # CutMix case - labels contains (labels_a, labels_b, lambda)
            labels_a, labels_b, lam = labels
            ce_loss = lam * self.ce_loss(cls_out, labels_a) + \
                     (1 - lam) * self.ce_loss(cls_out, labels_b)
        else:
            # Regular case
            ce_loss = self.ce_loss(cls_out, labels)
        
        # Distillation loss
        dist_loss = self.ce_loss(dist_out, teacher_labels)
        
        # Combined loss
        loss = self.alpha * ce_loss + (1 - self.alpha) * dist_loss
        
        return loss