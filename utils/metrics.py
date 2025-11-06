"""
Metrics and statistics utilities.
"""
import numpy as np
import matplotlib.pyplot as plt


def calculate_dataset_stats(dataset, class_names=['Cats', 'Dogs']):
    """
    Calculate and display statistics about the dataset.
    
    Args:
        dataset: PyTorch Dataset object
        class_names (list): List of class names
    
    Returns:
        tuple: (unique_labels, counts)
    """
    labels = []
    for _, label in dataset:
        labels.append(label)
    
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    print("Class Distribution:")
    for i, count in zip(unique, counts):
        class_name = class_names[i]
        percentage = (count / len(labels)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    return unique, counts


def plot_class_distribution(train_counts, test_counts, class_names=['Cats', 'Dogs']):
    """
    Plot class distribution for train and test sets.
    
    Args:
        train_counts (array): Training set class counts
        test_counts (array): Test set class counts
        class_names (list): List of class names
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(class_names, train_counts, color=['blue', 'orange'])
    ax1.set_title('Training Set')
    ax1.set_ylabel('Number of images')
    
    ax2.bar(class_names, test_counts, color=['blue', 'orange'])
    ax2.set_title('Test Set')
    ax2.set_ylabel('Number of images')
    
    plt.tight_layout()
    plt.show()
