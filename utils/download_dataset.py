"""
Download Cats vs Dogs dataset from Kaggle.

Lanciabile direttamente con: !python utils/download_dataset.py
"""
import os
# Rimosso argparse
import shutil


def download_dataset(output_dir):
    """
    Download the Cats vs Dogs dataset from Kaggle.
    
    Args:
        output_dir (str): Directory to save the dataset
    """
    try:
        # Importiamo kagglehub qui per dare messaggi di errore più specifici
        import kagglehub
    except ImportError:
        print("❌ Error: kagglehub not installed")
        print("Install it with: pip install kagglehub")
        return False
    
    print("\n" + "="*60)
    print("DOWNLOADING CATS VS DOGS DATASET FROM KAGGLE")
    print("="*60 + "\n")
    
    # Download dataset
    print("Downloading dataset (this may take a while)...")
    try:
        # Nota: questo percorso è fisso per il dataset Cats vs Dogs
        path = kagglehub.dataset_download("tongpython/cat-and-dog")
    except Exception as e:
        print(f"❌ Errore durante il download da Kaggle: {e}")
        print("Assicurati di aver configurato le tue credenziali Kaggle (file kaggle.json) in Colab.")
        return False
        
    print(f"\n✓ Dataset downloaded to temporary location: {path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy files to output directory 
    print(f"\n✓ Copying files to destination: {output_dir}...")
    
    # Questa parte gestisce la copia dei file dalla cartella temporanea alla cartella di progetto ./data
    for item in os.listdir(path):
        src = os.path.join(path, item)
        dst = os.path.join(output_dir, item)
        
        if os.path.exists(dst):
            print(f"  - Skipping {item} (already exists)")
        else:
            if os.path.isdir(src):
                shutil.copytree(src, dst)
                print(f"  - Copied {item}/")
            else:
                shutil.copy2(src, dst)
                print(f"  - Copied {item}")
    
    print("\n" + "="*60)
    print("DATASET READY IN PROJECT FOLDER!")
    print("="*60)
    # Questa stampa aiuta a debuggare il percorso 'KeyError: label'
    print(f"\nVerificare che i file siano in: {output_dir}/training_set/training_set")
    
    return True


def main():
    """Main function: definisce i parametri e avvia il download."""
    
    # === PARAMETRI DEFINITI INTERNAMENTE ===
    output_dir = './data'
    # ======================================
    
    success = download_dataset(output_dir)
    
    if success:
        print("\nPronto per il training. Eseguire:")
        print("  !python train.py")
    else:
        print("\n❌ Dataset download failed. Controlla l'errore precedente.")


if __name__ == '__main__':
    main()