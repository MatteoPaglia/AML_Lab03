"""
Validation script for image classification with VGG16.
Used to evaluate the model on the validation set during a training run.

Lanciabile con: !python validate.py
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import wandb 

# Import custom modules
from dataset import CustomImageDataset, create_annotations_csv
from models import create_vgg16_model, count_trainable_parameters
from utils import get_val_test_transforms

# La funzione validate è quella che esegue la valutazione
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


def main():
    """Main validation function."""
    
    # === DEFINIZIONE DEI PARAMETRI ===
    # Assicurati che questi parametri siano gli stessi usati in train.py!
    params = {
        # Data arguments
        'data_dir': './data',
        'train_dir': 'training_set/training_set',
        
        # Model arguments
        'num_classes': 2,
        'pretrained': True,
        'freeze_base': True, 
        
        # Validation arguments
        'batch_size': 128,
        'val_split': 0.2,
        'checkpoint_path': './checkpoints/best_model.pth', # Percorso del checkpoint da valutare
        
        # Wandb logging
        'use_wandb': True, 
        'wandb_project': 'vgg16-finetuning-simple'
    }
    # ==================================
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")

    # 1. Carica il modello
    print("\n✓ Creating model...")
    model = create_vgg16_model(
        num_classes=params['num_classes'],
        pretrained=params['pretrained'],
        freeze_base=params['freeze_base']
    )
    
    # Carica i pesi dal checkpoint
    if not os.path.exists(params['checkpoint_path']):
        print(f"ERRORE: Checkpoint non trovato in {params['checkpoint_path']}.")
        print("Eseguire prima train.py per salvare il modello.")
        return
        
    checkpoint = torch.load(params['checkpoint_path'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"✓ Modello caricato dal checkpoint dell'epoca {checkpoint['epoch']}")
    
    criterion = nn.CrossEntropyLoss()
    
    # 2. Prepara il Dataset di Validation
    train_path = os.path.join(params['data_dir'], params['train_dir'])
    
    if not os.path.exists('train_annotations.csv'):
        print("✓ Creating training annotations...")
        create_annotations_csv(train_path, 'train_annotations.csv')
        
    val_transform = get_val_test_transforms()
    
    # Usiamo il set di training per estrarre la validazione
    valid_dataset = CustomImageDataset(
        annotations_file='train_annotations.csv',
        img_dir=train_path,
        transform=val_transform
    )
    
    # Creazione dello split di validazione (deve corrispondere a quello di train.py)
    indices = list(range(len(valid_dataset)))
    split = int(np.floor(params['val_split'] * len(valid_dataset)))
    valid_sample = SubsetRandomSampler(indices[:split]) # Solo la parte di validation
    
    validloader = DataLoader(
        valid_dataset, 
        sampler=valid_sample,
        batch_size=params['batch_size'], 
        num_workers=4
    )
    
    print(f"  - Validation samples: {len(indices[:split])}")
    print("="*60)
    print("STARTING VALIDATION")
    print("="*60)
    
    # 3. Valutazione
    valid_loss, valid_acc = validate(model, validloader, device, criterion)
    
    # 4. Risultati
    print(f"VALIDATION COMPLETED:")
    print(f"  -> Validation Loss: {valid_loss:.4f}")
    print(f"  -> Validation Accuracy: {valid_acc:.4f}")
    print("="*60 + "\n")
    
    # Logga risultati finali su Wandb (opzionale)
    if params['use_wandb']:
        try:
            wandb.init(
                project=params['wandb_project'],
                name='validation_final',
                job_type='validation_final',
                reinit=True
            )
            wandb.log({
                'final_validation_loss': valid_loss,
                'final_validation_accuracy': valid_acc
            })
            wandb.finish()
        except NameError:
             # Se wandb non è stato inizializzato nel blocco main()
             pass


if __name__ == '__main__':
    main()