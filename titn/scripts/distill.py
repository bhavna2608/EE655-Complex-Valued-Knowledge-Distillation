#!/usr/bin/env python3
import argparse
from models.teacher_models import get_teacher_model
from training.distillation import DistillationWrapper
from utils.helpers import load_config
from models import TITN 

def distill_teacher(config):
    teacher = get_teacher_model(config)
    student = TITN(config)
    
    # Initialize distillation wrapper
    distiller = DistillationWrapper(teacher, student)
    
    # You would add custom distillation logic here
    # For example, specialized training loops or layer-wise distillation
    
    return distiller

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Knowledge Distillation')
    parser.add_argument('--config', required=True, help='config file path')
    args = parser.parse_args()
    
    config = load_config(args.config)
    distiller = distill_teacher(config)
    print("Distillation setup complete!")
