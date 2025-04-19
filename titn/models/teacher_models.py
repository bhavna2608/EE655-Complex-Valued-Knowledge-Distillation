import torch
import torchvision.models as models

def get_vit_model(config):
    """Get Vision Transformer model for comparison"""
    from .vit import ViT
    return ViT(config)

def get_teacher_model(config):
    """Instantiate teacher model based on config"""
    name = config.teacher["name"].lower()
    
    # Create model
    if name == "vgg16":
        model = models.vgg16(pretrained=True)
    if name == "vgg16_bn":
        model = models.vgg16_bn(pretrained=True)
    elif name == "vgg19":
        model = models.vgg19(pretrained=True)
    elif name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif name == "densenet121":
        model = models.densenet121(pretrained=True)
    elif name == "mobilenet_v3":
        model = models.mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError(f"Unknown teacher model: {name}")
    
    # Modify for CIFAR (if needed)
    if config.dataset in ["cifar10", "cifar100"]:
        if hasattr(model, "classifier"):
            if isinstance(model.classifier, torch.nn.Sequential):
                # For VGG models
                model.classifier[-1] = torch.nn.Linear(
                    model.classifier[-1].in_features, 
                    config.num_classes
                )
            else:
                # For DenseNet, MobileNetV3, etc. where classifier is a Linear layer
                model.classifier = torch.nn.Linear(
                    model.classifier.in_features,
                    config.num_classes
                )
        elif hasattr(model, "fc"):
            # For ResNet
            model.fc = torch.nn.Linear(model.fc.in_features, config.num_classes)
    
    # Load custom weights if specified
    if config.teacher["weights"]:
        state_dict = torch.load(config.teacher["weights"])
        model.load_state_dict(state_dict)
    
    # Freeze parameters if needed
    if config.teacher["freeze"]:
        for param in model.parameters():
            param.requires_grad = False
    
    return model.eval()