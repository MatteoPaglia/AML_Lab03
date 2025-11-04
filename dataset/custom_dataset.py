"""
Custom Dataset implementation for image classification tasks.
Based on MLVM Lab3 warmup notebook.
"""
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class CustomImageDataset(Dataset):
    """
    Custom Dataset for loading images from a CSV annotation file.
    
    Args:
        annotations_file (str): Path to the CSV file with annotations (filename, label)
        img_dir (str): Directory containing the images
        transform (callable, optional): Optional transform to be applied on images
        target_transform (callable, optional): Optional transform to be applied on labels
    """
    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get image path
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        
        # Load image using PIL (more compatible)
        image = Image.open(img_path).convert('RGB')
        
        # Get label
        label = self.img_labels.iloc[idx, 1]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label


def create_annotations_csv(data_dir, output_csv, class_folders=None):
    """
    Create a CSV file with annotations from a directory structure.
    
    Args:
        data_dir (str): Root directory containing class folders
        output_csv (str): Output path for the CSV file
        class_folders (dict, optional): Mapping of folder names to class labels
                                       Default: {'cats': 0, 'dogs': 1}
    
    Returns:
        pd.DataFrame: The created annotations dataframe
    """
    if class_folders is None:
        class_folders = {'cats': 0, 'dogs': 1}
    
    data = []
    
    for folder_name, label in class_folders.items():
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({
                        'filename': os.path.join(folder_name, img_name),
                        'label': label
                    })
    
    # Create DataFrame and save as CSV
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    
    print(f"Created {output_csv} with {len(df)} images")
    for folder_name, label in class_folders.items():
        count = len(df[df['label'] == label])
        print(f"  {folder_name.capitalize()}: {count}")
    
    return df
