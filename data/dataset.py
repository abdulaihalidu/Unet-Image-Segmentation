import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import (RandomHorizontalFlip, RandomVerticalFlip,
                                    RandomResizedCrop, ToTensor, Normalize)
import elasticdeform as etorch
import random
from torchvision.transforms import functional as F
from pathlib import Path
import cv2
import hdf5storage


def elastic_deformation(image, mask=None, sigma=10, points=3):
    """
    Apply elastic deformation to an image (and optional mask).

    Parameters:
        image (PIL Image or np.ndarray): The input image.
        mask (PIL Image or np.ndarray, optional): The corresponding mask.
        sigma (float): Standard deviation of the Gaussian distribution for displacement.
        points (int): Number of control points for the coarse grid.

    Returns:
        Deformed image (and mask if provided).
    """
    # Convert PIL images to numpy arrays if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    if mask is not None and isinstance(mask, Image.Image):
        mask = np.array(mask)

    # Prepare the input for elasticdeform
    inputs = [image]
    if mask is not None:
        inputs.append(mask)

    # Apply elastic deformation
    deformed = etorch.deform_random_grid(inputs, sigma=sigma, points=points, order=[3, 0])

    # Convert back to PIL images
    deformed_image = Image.fromarray(deformed[0])
    if mask is not None:
        deformed_mask = Image.fromarray(deformed[1])
        return deformed_image, deformed_mask
    return deformed_image

  

class CellDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, test_mode=False):
        """
        Args:
            root_dir (str): Path to the dataset directory
            split (str): Dataset split ('train', 'val', 'test')
            transform (callable, optional): Transform to apply
            test_mode (bool): If True, only loads images
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.test_mode = test_mode

        # Load and validate files
        imgs_dir = os.path.join(root_dir, split, 'imgs')
        labels_dir = os.path.join(root_dir, split, 'labels')
        self.image_files = sorted([f for f in os.listdir(imgs_dir) if f.endswith(('png', 'jpg', 'jpeg'))])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith(('png', 'jpg', 'jpeg'))]) if not test_mode else []

        if not test_mode and len(self.image_files) != len(self.label_files):
            raise ValueError("Number of images and labels do not match.")
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.split, 'imgs', self.image_files[idx])
        image = Image.open(img_path).convert('L')  # Load as grayscale

        if self.test_mode:
            # The dummy mask is just to avoid problems using the same dataclass for train/test set
            dummy_mask = torch.zeros((1, image.size[1], image.size[0]), dtype=torch.float32)
            # Just apply min-max normalization to image if transform is not provided
            image = self.transform(image, None)[0] if self.transform \
                else torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
            file_name = self.image_files[idx]
            return image, file_name, dummy_mask

        label_path = os.path.join(self.root_dir, self.split, 'labels', self.label_files[idx])
        mask = Image.open(label_path).convert('L')

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = np.array(image).astype(np.float32) / 255.0
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)  # Binarize and add channel dim

        return image, mask

class DualTransform:
    """Applies the same transform to both image and label."""
    def __init__(self, transforms, apply_elastic=False, elastic_params=None):
        self.transforms = transforms
        self.apply_elastic = apply_elastic
        self.elastic_params = elastic_params if elastic_params else {'sigma': 10, 'points': 3}

    def __call__(self, image, label=None):
        if image is None:
            raise ValueError("Image cannot be None.")

        # Apply elastic deformation if specified
        if self.apply_elastic:
            image, label = elastic_deformation(image, label, **self.elastic_params)

        # Apply other transformations
        for transform in self.transforms:
            if isinstance(transform, (RandomHorizontalFlip, RandomVerticalFlip)) and random.random() < transform.p:
                flip_func = F.hflip if isinstance(transform, RandomHorizontalFlip) else F.vflip
                image = flip_func(image)
                if label:
                    label = flip_func(label)
            elif isinstance(transform, RandomResizedCrop):
                i, j, h, w = transform.get_params(image, transform.scale, transform.ratio)
                image = F.resized_crop(image, i, j, h, w, transform.size, transform.interpolation)
                if label:
                    label = F.resized_crop(label, i, j, h, w, transform.size, F.InterpolationMode.NEAREST)
            elif isinstance(transform, ToTensor):
                image = F.to_tensor(image)
                if label:
                    label = F.to_tensor(label)
            elif isinstance(transform, Normalize):
                image = F.normalize(image, transform.mean, transform.std)

        return image, label


################################ Brain Tumor Dataset ###################################
class BrainTumorDataset(Dataset):
    """
    Dataset for Brain Tumor Images and Masks.
    """

    def __init__(self, data_dir, image_dimension=128, transform=None):
        self.data_dir = Path(data_dir).resolve()
        self.image_dimension = image_dimension
        self.transform = transform

        # Lazy loading metadata to optimize memory usage
        self.files = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.mat'])
        if not self.files:
            raise FileNotFoundError(f"No MATLAB files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        # Load the .mat file on demand
        mat_file = hdf5storage.loadmat(str(file))['cjdata'][0]

        # Process image and mask lazily
        image = cv2.resize(mat_file[2], (self.image_dimension, self.image_dimension), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mat_file[4].astype('uint8'), (self.image_dimension, self.image_dimension), interpolation=cv2.INTER_CUBIC)

        # Normalize image and add channel dimension
        image = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)  # Avoid divide-by-zero
        image = image[np.newaxis, ...].astype(np.float32)  # Add channel dimension

        mask = mask[np.newaxis, ...].astype(np.float32)  # Add channel dimension

        # Apply optional transformations
        if self.transform:
            image, mask = self.transform(image, mask)

        return torch.tensor(image), torch.tensor(mask)


def get_dataloaders(data_dir, image_dimension=512, batch_size=2, val_split=0.15, test_split=0.15, num_workers=0):
    """
    Splits dataset and creates PyTorch DataLoaders with memory-efficient lazy loading.

    Args:
        data_dir (str or Path): Path to the dataset directory.
        image_dimension (int): Dimension to resize images and masks.
        batch_size (int): Batch size for DataLoader.
        val_split (float): Fraction of data for validation.
        test_split (float): Fraction of data for testing.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    dataset = BrainTumorDataset(data_dir, image_dimension)

    # Calculate split sizes
    test_size = int(len(dataset) * test_split)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - test_size - val_size

    # Create train, validation, and test splits
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(3)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader