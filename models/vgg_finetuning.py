"""
VGG16 model for fine-tuning on custom classification tasks.
Based on MLVM Lab3 warmup notebook.
"""
import torch
import torch.nn as nn
from torchvision import models


def create_vgg16_model(num_classes=2, pretrained=True, freeze_base=True):
    """
    Create a VGG16 model for fine-tuning.
    
    Args:
        num_classes (int): Number of output classes. Default: 2 (binary classification)
        pretrained (bool): Whether to use pre-trained ImageNet weights. Default: True
        freeze_base (bool): Whether to freeze the base convolutional layers. Default: True
                           If True, only the final classifier layer will be trainable (Feature Extraction).
                           If False, the entire network can be fine-tuned.
    
    Returns:
        torch.nn.Module: The modified VGG16 model
    """
    # Load pre-trained VGG16
    if pretrained:
        model = models.vgg16(weights='IMAGENET1K_V1')
    else:
        model = models.vgg16(weights=None)
    
    # Replace the final classifier layer
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    
    # Freeze base layers if requested (Feature Extraction mode)
    if freeze_base:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final classifier layer
        for param in model.classifier[6].parameters():
            param.requires_grad = True
    
    return model


def unfreeze_layers(model, num_layers=1):
    """
    Unfreeze the last N layers of the classifier.
    Useful for progressive fine-tuning.
    
    Args:
        model: The VGG16 model
        num_layers (int): Number of classifier layers to unfreeze (starting from the end)
    
    Returns:
        torch.nn.Module: The model with unfrozen layers
    """
    # First freeze all
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last N layers of classifier
    classifier_layers = list(model.classifier.children())
    for i in range(num_layers):
        layer_idx = -(i + 1)  # Start from the end
        for param in classifier_layers[layer_idx].parameters():
            param.requires_grad = True
    
    return model


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        tuple: (trainable_params, total_params)
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params
