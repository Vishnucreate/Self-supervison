from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import random
import torch

def compute_train_transform(seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    Retain normalization and implement the required augmentations.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Transformation that applies color jitter with brightness=0.4, contrast=0.4, saturation=0.4, and hue=0.1
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
    
    train_transform = transforms.Compose([
        # Step 1: Randomly resize and crop to 32x32.
        transforms.RandomResizedCrop(32),

        # Step 2: Horizontally flip the image with probability 0.5
        transforms.RandomHorizontalFlip(p=0.5),

        # Step 3: With a probability of 0.8, apply color jitter
        transforms.RandomApply([color_jitter], p=0.8),

        # Step 4: With a probability of 0.2, convert the image to grayscale
        transforms.RandomGrayscale(p=0.2),

        # Convert to tensor and normalize using CIFAR10 statistics
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return train_transform

def compute_test_transform():
    """
    Test data transformation: Only normalization (no augmentations).
    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_transform


class CIFAR10Pair(CIFAR10):
    """
    CIFAR10 Dataset returning a pair of augmented views for self-supervised learning.
    """
    def __getitem__(self, index):
        # Load image and target
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # Apply the same transform to generate two augmented views
        x_i = None
        x_j = None

        if self.transform is not None:
            x_i = self.transform(img)  # First augmented view
            x_j = self.transform(img)  # Second augmented view

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return the pair of augmented views and the target
        return x_i, x_j, target
