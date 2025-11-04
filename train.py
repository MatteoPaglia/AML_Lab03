"""
Training script for image classification with VGG16.
Based on MLVM Lab3 warmup notebook - structured version.

Usage:
    python train.py --data_dir <path> --epochs 10 --batch_size 128 --lr 0.0001
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Import custom modules
from dataset import CustomImageDataset, create_annotations_csv
from models import create_vgg16_model, count_trainable_parameters
from utils import get_train_transforms, get_val_test_transforms, plot_training_history


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train VGG16 for image classification')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--train_dir', type=str, default='training_set/training_set',
                       help='Relative path to training set')
    parser.add_argument('--test_dir', type=str, default='test_set/test_set',
                       help='Relative path to test set')
    
    # Model arguments
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of output classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ImageNet weights')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                       help='Freeze base layers (feature extraction mode)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (0.2 = 20%%)')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    
    # Wandb logging
    parser.add_argument('--use_wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='vgg16-finetuning',
                       help='Wandb project name')
    
    return parser.parse_args()


def train_one_epoch(model, trainloader, device, optimizer, criterion):
    """
    Train for one epoch.
    
    Args:
        model: PyTorch model
        trainloader: DataLoader for training
        device: Device to run on
        optimizer: Optimizer
        criterion: Loss function
    
    Returns:
        float: Average training loss
    """
    model.train()
    running_loss = 0.0
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(trainloader)


def validate(model, validloader, device, criterion):
    """
    Validate the model.
    
    Args:
        model: PyTorch model
        validloader: DataLoader for validation
        device: Device to run on
        criterion: Loss function
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in validloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(validloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


def train(model, trainloader, validloader, device, optimizer, criterion, args, wandb_run=None):
    """
    Main training loop following Lab02 style.
    
    Args:
        model: PyTorch model
        trainloader: DataLoader for training
        validloader: DataLoader for validation
        device: Device to run on
        optimizer: Optimizer
        criterion: Loss function
        args: Command line arguments
        wandb_run: Wandb run object (optional)
    
    Returns:
        tuple: (train_losses, valid_losses, valid_accuracies)
    """
    train_losses = []
    valid_losses = []
    valid_accuracies = []
    best_val_loss = float('inf')
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(args.epochs):
        # Training
        train_loss = train_one_epoch(model, trainloader, device, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Validation
        valid_loss, valid_acc = validate(model, validloader, device, criterion)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | "
              f"Valid Loss: {valid_loss:.4f} | "
              f"Valid Acc: {valid_acc:.4f}")
        
        # 3. Log metrics over time to visualize performance (slide style)
        if wandb_run:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_acc,
                'learning_rate': optimizer.param_groups[0]['lr']  # Log current LR
            })
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or valid_loss < best_val_loss:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_acc,
            }, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
            
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'valid_accuracy': valid_acc,
                }, best_path)
                print(f"  ✓ New best model saved: {best_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print("="*60 + "\n")
    
    return train_losses, valid_losses, valid_accuracies


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize wandb if requested (following professor's slide style)
    wandb_run = None
    if args.use_wandb:
        import wandb
        # 1. Start a W&B run (slide style)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=f'vgg16-lr{args.lr}-bs{args.batch_size}',
            tags=['vgg16', 'transfer-learning', 'cats-vs-dogs']
        )
        
        # 2. Save model inputs and hyperparameters (slide style)
        config = wandb.config
        config.learning_rate = args.lr
        config.batch_size = args.batch_size
        config.epochs = args.epochs
        config.momentum = args.momentum
        config.val_split = args.val_split
        config.architecture = 'VGG16'
        config.num_classes = args.num_classes
        config.mode = 'feature_extraction' if args.freeze_base else 'full_finetuning'
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Prepare dataset paths
    train_path = os.path.join(args.data_dir, args.train_dir)
    test_path = os.path.join(args.data_dir, args.test_dir)
    
    # Create annotation files if they don't exist
    if not os.path.exists('train_annotations.csv'):
        print("\n✓ Creating training annotations...")
        create_annotations_csv(train_path, 'train_annotations.csv')
    
    if not os.path.exists('test_annotations.csv'):
        print("✓ Creating test annotations...")
        create_annotations_csv(test_path, 'test_annotations.csv')
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_test_transforms()
    
    # Create datasets
    print("\n✓ Loading datasets...")
    train_dataset = CustomImageDataset(
        annotations_file='train_annotations.csv',
        img_dir=train_path,
        transform=train_transform
    )
    
    valid_dataset = CustomImageDataset(
        annotations_file='train_annotations.csv',
        img_dir=train_path,
        transform=val_transform
    )
    
    # Create train/validation split
    indices = list(range(len(train_dataset)))
    split = int(np.floor(args.val_split * len(train_dataset)))
    train_sample = SubsetRandomSampler(indices[split:])
    valid_sample = SubsetRandomSampler(indices[:split])
    
    # Create dataloaders
    trainloader = DataLoader(train_dataset, sampler=train_sample, 
                            batch_size=args.batch_size, num_workers=4)
    validloader = DataLoader(valid_dataset, sampler=valid_sample, 
                            batch_size=args.batch_size, num_workers=4)
    
    print(f"  - Training samples: {len(indices[split:])}")
    print(f"  - Validation samples: {len(indices[:split])}")
    print(f"  - Batch size: {args.batch_size}")
    
    # Create model
    print("\n✓ Creating model...")
    model = create_vgg16_model(
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        freeze_base=args.freeze_base
    )
    model = model.to(device)
    
    trainable, total = count_trainable_parameters(model)
    print(f"  - Trainable parameters: {trainable:,} / {total:,}")
    print(f"  - Mode: {'Feature Extraction' if args.freeze_base else 'Full Fine-tuning'}")
    
    # Watch model with wandb (log gradients and parameters)
    if wandb_run:
        wandb_run.watch(model, log='all', log_freq=100)
        print("  - Wandb watching model gradients")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters_to_optimize, lr=args.lr, momentum=args.momentum)
    
    print("\n✓ Training setup:")
    print(f"  - Loss: CrossEntropyLoss")
    print(f"  - Optimizer: SGD (lr={args.lr}, momentum={args.momentum})")
    print(f"  - Epochs: {args.epochs}")
    
    # Train
    train_losses, valid_losses, valid_accuracies = train(
        model, trainloader, validloader, device, optimizer, criterion, args, wandb_run
    )
    
    # Plot training history
    plot_training_history(train_losses, valid_losses, 
                         title='Training History - Feature Extraction')
    
    # Finish wandb
    if wandb_run:
        wandb_run.finish()
    
    print("\n✓ Training completed! Checkpoints saved in:", args.checkpoint_dir)


if __name__ == '__main__':
    main()
