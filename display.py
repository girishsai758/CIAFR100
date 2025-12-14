import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
# Define the root directory where the data will be downloaded/loaded
DATA_ROOT = './cifar100_data'
# --- End Configuration ---

def load_and_inspect_cifar100(root_dir):
    """
    Loads the CIFAR-100 dataset, inspects its structure, and displays sample images.
    """
    print("--- 1. Data Loading and Download Check ---")
    
    # Define transformations: Convert to Tensor and Normalize
    # Normalization is crucial for training, even for inspection
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
#     transforms.ToTensor():
# This transformation converts an image from a PIL (Pillow) Image or a NumPy array into a PyTorch Tensor.
# It also automatically scales the pixel values from the typical range of [0, 255] (for 8-bit images) to [0.0, 1.0]. 
# This normalization is important because neural networks generally perform better with input values within a smaller, consistent range.
# Additionally, it rearranges the dimensions of the image tensor from (Height, Width, Channels) to (Channels, Height, Width), which is the standard format expected by PyTorch convolutional layers.
    # Load the training dataset (this triggers the download if not present)
    try:
        train_data = datasets.CIFAR100(
            root=root_dir, 
            download=False, 
            train=True, 
            transform=transform
        )
        test_data = datasets.CIFAR100(
            root=root_dir, 
            download=False, 
            train=False, 
            transform=transform
        )
    except Exception as e:
        print(f"Error loading or downloading CIFAR-100: {e}")
        print("Please ensure you have an active internet connection for the initial download.")
        return

    print(f"Train Dataset Size: {len(train_data)} images")
    print(f"Test Dataset Size: {len(test_data)} images")

    # --- 2. Inspecting Data Dimensions and Classes ---
    
    # Get the image and label of the first sample
    sample_image, sample_label_idx = train_data[0] 
    
    print("\n--- 2. Data Structure Inspection ---")
    print(f"Sample Image Tensor Shape (C, H, W): {sample_image.shape}") # Should be (3, 32, 32)
    print(f"Data Type: {sample_image.dtype}")
    print(f"Number of Classes (Total): {len(train_data.classes)}") # Should be 100
    
    # The actual class names
    class_names = train_data.classes

    # --- 3. Visualization: Displaying a Grid of Sample Images ---
    
    print("\n--- 3. Visualizing Sample Data ---")
    
    # We will sample one image from each of the 100 classes for a 10x10 grid.
    
    # Find one index for each class
    class_indices = {}
    for i in range(len(train_data)):
        label = train_data[i][1]
        if label not in class_indices:
            class_indices[label] = i
        if len(class_indices) == 100:
            break

    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    axes = axes.flatten()

    for i, (label_index, data_index) in enumerate(class_indices.items()):
        # Get the normalized image tensor and label index
        img_tensor, _ = train_data[data_index]
        class_name = class_names[label_index]

        # Denormalize the image tensor to display correctly
        # Formula: Image = (Normalized_Image * std) + mean
        # Since we used (0.5, 0.5, 0.5) for both mean and std:
        img_array = img_tensor.numpy() / 2.0 + 0.5 # Convert from [-1, 1] to [0, 1]
        
        # Convert from (C, H, W) to (H, W, C) for Matplotlib
        img_array = np.transpose(img_array, (1, 2, 0)) 
        
        # Display
        axes[i].imshow(img_array)
        axes[i].set_title(class_name, fontsize=8)
        axes[i].axis('off')

    plt.suptitle("CIFAR-100: Sample Image for Each of the 100 Fine-Grained Classes", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Execute the inspection function
if __name__ == "__main__":
    load_and_inspect_cifar100(DATA_ROOT)