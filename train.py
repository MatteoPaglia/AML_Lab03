"""
Training script for image classification with VGG16.
Based on MLVM Lab3 warmup notebook - structured version.

Lanciabile direttamente con: !python train.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Import custom modules
from dataset import CustomImageDataset, create_annotations_csv
from models import create_vgg16_model
from utils import get_train_transforms, get_val_test_transforms, plot_training_history

# Rimuoviamo la funzione parse_args() e la classe Config

# Le funzioni train_one_epoch, validate e train non cambiano
# tranne per il fatto che 'args' viene sostituito con 'params' nel contesto

def train_one_epoch(model, trainloader, device, optimizer, criterion):
    """
    Train for one epoch.
    ... (corpo della funzione invariato) ...
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
    ... (corpo della funzione invariato) ...
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



def train(model, trainloader, device, optimizer, criterion, params, wandb_run=None):
    """
    Main training loop (TRAIN ONLY).
    """
    train_losses = []
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    for epoch in range(params['epochs']):
        # Training
        train_loss = train_one_epoch(model, trainloader, device, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{params['epochs']}] Train Loss: {train_loss:.4f}")
        
        # 3. Log metrics over time (SOLO TRAIN LOSS)
        if wandb_run:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save checkpoint (SALVA SEMPRE ALLA FINE DI OGNI EPOCA O SECONDO LOGICA)
        if (epoch + 1) % params['save_every'] == 0 or (epoch + 1) == params['epochs']:
            checkpoint_path = os.path.join(params['checkpoint_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            # Salviamo il modello finale come "best_model.pth" per l'eval/validate
            if (epoch + 1) == params['epochs']:
                best_path = os.path.join(params['checkpoint_dir'], 'best_model.pth')
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Final model saved as best_model.pth: {best_path}")
            
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
            
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60 + "\n")
    
    return train_losses


def main():
    """Main training function."""
    
    # === DEFINIZIONE DEI PARAMETRI (al posto di argparse) ===
    params = {
        # Data arguments
        'data_dir': './data',
        'train_dir': 'training_set/training_set',
        'test_dir': 'test_set/test_set',
        
        # Model arguments
        'num_classes': 2,
        'pretrained': True,
        'freeze_base': True, # Feature extraction mode
        
        # Training arguments
        'epochs': 10,
        'batch_size': 128,
        'lr': 0.0001,
        'momentum': 0.9,
        'val_split': 0.2,
        
        # Output arguments
        'checkpoint_dir': './checkpoints',
        'save_every': 5,
        
        # Wandb logging
        'use_wandb': True, # Abilitato per il Lab3
        'wandb_project': 'vgg16-finetuning-simple'
    }
    # =======================================================
    
    # Initialize wandb if requested
    wandb_run = None
    if params['use_wandb']:
        import wandb
        # 1. Start a W&B run
        wandb_run = wandb.init(
            project=params['wandb_project'],
            name=f"vgg16-lr{params['lr']}-bs{params['batch_size']}",
            tags=['vgg16', 'transfer-learning', 'cats-vs-dogs']
        )
        
        # 2. Save model inputs and hyperparameters
        wandb_run.config.update({
            'learning_rate': params['lr'],
            'batch_size': params['batch_size'],
            'epochs': params['epochs'],
            'momentum': params['momentum'],
            'val_split': params['val_split'],
            'architecture': 'VGG16',
            'num_classes': params['num_classes'],
            'mode': 'feature_extraction' if params['freeze_base'] else 'full_finetuning'
        })
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(params['checkpoint_dir'], exist_ok=True)
    
    # Prepare dataset paths
    train_path = os.path.join(params['data_dir'], params['train_dir'])
    test_path = os.path.join(params['data_dir'], params['test_dir'])
    
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
    split = int(np.floor(params['val_split'] * len(train_dataset)))
    train_sample = SubsetRandomSampler(indices[split:])
    valid_sample = SubsetRandomSampler(indices[:split])
    
    # Create dataloaders
    trainloader = DataLoader(train_dataset, sampler=train_sample, 
                            batch_size=params['batch_size'], num_workers=4)
    validloader = DataLoader(valid_dataset, sampler=valid_sample, 
                            batch_size=params['batch_size'], num_workers=4)
    
    print(f"  - Training samples: {len(indices[split:])}")
    print(f"  - Validation samples: {len(indices[:split])}")
    print(f"  - Batch size: {params['batch_size']}")
    
    # Create model
    print("\n✓ Creating model...")
    model = create_vgg16_model(
        num_classes=params['num_classes'],
        pretrained=params['pretrained'],
        freeze_base=params['freeze_base']
    )
    model = model.to(device)
    
    
    print(f"  - Mode: {'Feature Extraction' if params['freeze_base'] else 'Full Fine-tuning'}")
    
    # Watch model with wandb (log gradients and parameters)
    if wandb_run:
        wandb_run.watch(model, log='all', log_freq=100)
        print("  - Wandb watching model gradients")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters_to_optimize, lr=params['lr'], momentum=params['momentum'])
    
    print("\n✓ Training setup:")
    print(f"  - Loss: CrossEntropyLoss")
    print(f"  - Optimizer: SGD (lr={params['lr']}, momentum={params['momentum']})")
    print(f"  - Epochs: {params['epochs']}")
    
    # Train
    train_losses, valid_losses, valid_accuracies = train(
        model, trainloader, validloader, device, optimizer, criterion, params, wandb_run
    )
    
    # Plot training history
    plot_training_history(train_losses, valid_losses, 
                         title='Training History - Feature Extraction')
    
    # Finish wandb
    if wandb_run:
        wandb_run.finish()
    
    print("\n✓ Training completed! Checkpoints saved in:", params['checkpoint_dir'])


if __name__ == '__main__':
    main()