"""
Data transforms for training and evaluation.
Following ImageNet normalization and best practices from Lab3.
"""
from torchvision import transforms


def get_train_transforms(image_size=224):
    """
    Get transforms for TRAINING SET with data augmentation.
    
    Args:
        image_size (int): Target image size. Default: 224
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),  # Augmentation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_test_transforms(image_size=224):
    """
    Get transforms for VALIDATION and TEST sets WITHOUT augmentation.
    
    IMPORTANT: No random augmentation for consistent evaluation!
    
    Args:
        image_size (int): Target image size. Default: 224
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),  # Deterministic center crop
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])
