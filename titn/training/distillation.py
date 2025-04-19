import torch
import torch.nn as nn

class DistillationWrapper(nn.Module):
    """Handles knowledge distillation logic"""
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        student_output = self.student(x)
        return teacher_output, student_output
    
    def get_learnable_params(self):
        """Returns only student parameters for optimizer"""
        return self.student.parameters()
    
    def get_soft_labels(self, x):
        """Get teacher's soft predictions"""
        with torch.no_grad():
            return self.teacher(x)