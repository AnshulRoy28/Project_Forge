import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

class DataLoadingError(Exception):
    """Custom exception for data loading issues."""
    def __init__(self, message: str, fix: str):
        self.message = message
        self.fix = fix
        super().__init__(message)

def get_mnist_data_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST dataset, applies transformations, and creates PyTorch DataLoaders.

    Args:
        data_root (str): The root directory where the MNIST dataset will be stored.
                         This should be '/data' inside the Docker container.
        batch_size (int): The number of samples per batch.
        num_workers (int): How many subprocesses to use for data loading. 0 means that
                           the data will be loaded in the main process.
        pin_memory (bool): If True, the data loader will copy Tensors into device
                           (CUDA) pinned memory before returning them.

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing (train_loader, test_loader).

    Raises:
        DataLoadingError: If there's an issue with downloading or loading the dataset.
    """
    if not os.path.exists(data_root):
        logger.warning(f"Data root directory '{data_root}' does not exist. Attempting to create.")
        try:
            os.makedirs(data_root, exist_ok=True)
            logger.info(f"Created data root directory: {data_root}")
        except OSError as e:
            logger.error(f"Failed to create data root directory '{data_root}': {e}", exc_info=True)
            raise DataLoadingError(
                f"Cannot create data root directory '{data_root}'.",
                f"Ensure the path is valid and you have write permissions. Error: {e}"
            )

    # Define standard transformations for MNIST
    # Convert images to PyTorch Tensors and normalize pixel values
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST mean and standard deviation
    ])

    try:
        # Load the training dataset
        # If not present, it will be downloaded to data_root
        train_dataset = datasets.MNIST(
            root=data_root,
            train=True,
            download=True,
            transform=transform
        )
        logger.info(f"MNIST training dataset loaded. Number of samples: {len(train_dataset)}")

        # Load the test dataset
        test_dataset = datasets.MNIST(
            root=data_root,
            train=False,
            download=True,
            transform=transform
        )
        logger.info(f"MNIST test dataset loaded. Number of samples: {len(test_dataset)}")

    except Exception as e:
        logger.error(f"Failed to download or load MNIST dataset: {e}", exc_info=True)
        raise DataLoadingError(
            "Failed to download or load MNIST dataset.",
            f"Check your internet connection and ensure '{data_root}' is writable. Error: {e}"
        )

    try:
        # Create DataLoaders for batching and shuffling
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        logger.info(f"DataLoaders created with batch_size={batch_size}, num_workers={num_workers}.")
        return train_loader, test_loader

    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {e}", exc_info=True)
        raise DataLoadingError(
            "Failed to create PyTorch DataLoaders.",
            f"Check batch_size and num_workers parameters. Error: {e}"
        )