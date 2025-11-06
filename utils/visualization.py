"""
Visualization utilities for images and training progress.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize a tensor image with mean and standard deviation.
    
    Args:
        tensor (torch.Tensor): Normalized image tensor
        mean (list): Mean used for normalization
        std (list): Std used for normalization
    
    Returns:
        torch.Tensor: Denormalized tensor
    """
    tensor_copy = tensor.clone()
    for t, m, s in zip(tensor_copy, mean, std):
        t.mul_(s).add_(m)
    return tensor_copy


def visualize_batch(images, labels, class_names=['Cat', 'Dog'], num_images=8):
    """
    Visualize a batch of images with their labels.
    
    Args:
        images (torch.Tensor): Batch of images [B, C, H, W]
        labels (torch.Tensor): Batch of labels [B]
        class_names (list): List of class names
        num_images (int): Number of images to display
    """
    num_images = min(num_images, len(images))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx in range(num_images):
        # Denormalize image
        img = denormalize(images[idx].clone())
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        # Display
        axes[idx].imshow(img)
        label_name = class_names[labels[idx].item()]
        axes[idx].set_title(f"{label_name}")
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, valid_losses, title='Training History'):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (list): List of training losses per epoch
        valid_losses (list): List of validation losses per epoch
        title (str): Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(valid_losses, label='Valid Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
