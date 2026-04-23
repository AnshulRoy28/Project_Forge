import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
import sys
import time
import argparse
import logging
from datetime import datetime

# Import model and dataset modules
from model import FeedForwardNN
from dataset import get_mnist_data_loaders, DataLoadingError
from torch.utils.data import DataLoader

# --- Setup Logging ---
def setup_logging(log_dir: str, log_file_name: str = "training.log"):
    """Configures logging for the training script."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Set higher logging level for some libraries to reduce verbosity
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("torchvision").setLevel(logging.INFO)
    return logging.getLogger(__name__)

logger = setup_logging("/workspace/logs") # Initialize with default path, will be updated by config

# --- Custom Exceptions ---
class ConfigError(Exception):
    """Custom exception for configuration loading issues."""
    def __init__(self, message: str, fix: str):
        self.message = message
        self.fix = fix
        super().__init__(message)

class TrainingError(Exception):
    """Custom exception for general training issues."""
    def __init__(self, message: str, fix: str = "Review logs for detailed error and suggested fixes."):
        self.message = message
        self.fix = fix
        super().__init__(message)

# --- Configuration Loading ---
def load_config(config_path: str) -> dict:
    """
    Loads configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Loaded configuration.

    Raises:
        ConfigError: If the config file cannot be found or parsed.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at: {config_path}")
        raise ConfigError(
            f"Configuration file '{config_path}' not found.",
            "Ensure 'config.yaml' exists in the /workspace directory or provide the correct path."
        )
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file '{config_path}': {e}", exc_info=True)
        raise ConfigError(
            f"Failed to parse configuration file '{config_path}'.",
            f"Check YAML syntax for errors. Error: {e}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config '{config_path}': {e}", exc_info=True)
        raise ConfigError(
            f"An unexpected error occurred while loading config '{config_path}'.",
            f"Ensure the file is readable and valid. Error: {e}"
        )

# --- Training and Validation Functions ---
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple[float, float]:
    """
    Performs one epoch of training.

    Returns:
        tuple[float, float]: Average training loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

            if batch_idx % 100 == 0:
                logger.info(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        except Exception as e:
            logger.error(f"Error during training batch {batch_idx} of epoch {epoch}: {e}", exc_info=True)
            raise TrainingError(
                f"Training failed during batch {batch_idx} of epoch {epoch}.",
                f"Review model, data, and optimizer setup. Error: {e}"
            )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    logger.info(f"Epoch {epoch} Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy

def validate_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> tuple[float, float]:
    """
    Performs one epoch of validation.

    Returns:
        tuple[float, float]: Average validation loss and accuracy for the epoch.
    """
    model.eval()
    validation_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                validation_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total_samples += target.size(0)
                correct_predictions += (predicted == target).sum().item()



            except Exception as e:
                logger.error(f"Error during validation batch {batch_idx} of epoch {epoch}: {e}", exc_info=True)
                raise TrainingError(
                    f"Validation failed during batch {batch_idx} of epoch {epoch}.",
                    f"Review model and data setup. Error: {e}"
                )

    epoch_loss = validation_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    logger.info(f"Epoch {epoch} Validation - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
    return epoch_loss, epoch_accuracy

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    accuracy: float,
    checkpoint_dir: str,
    best_accuracy: float,
    is_best: bool
):
    """
    Saves the model and optimizer state as a checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'best_accuracy': best_accuracy
    }
    try:
        torch.save(state, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path} (Accuracy: {accuracy:.4f})")
        if is_best:
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(state, best_model_path)
            logger.info(f"Best model saved to {best_model_path} (Accuracy: {accuracy:.4f})")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}", exc_info=True)
        raise TrainingError(
            f"Failed to save checkpoint for epoch {epoch}.",
            f"Ensure '{checkpoint_dir}' is writable and there is enough disk space. Error: {e}"
        )

def main():
    """
    Main function to orchestrate the training process.
    """
    parser = argparse.ArgumentParser(description="Train a FeedForwardNN on MNIST.")
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/config.yaml",
        help="Path to the configuration YAML file."
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        # Reconfigure logger with paths from config if needed, though /workspace/logs is fixed
        global logger
        logger = setup_logging(config['training_params']['log_dir'])
        logger.info("Starting MNIST FeedForwardNN training.")
        logger.debug(f"Loaded configuration: {config}")

        # --- Device Configuration ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

        # --- Data Loading ---
        data_root = config['data_params']['data_root']
        batch_size = config['data_params']['batch_size']
        train_loader, test_loader = get_mnist_data_loaders(data_root, batch_size)
        logger.info(f"Number of training batches: {len(train_loader)}")
        logger.info(f"Number of test batches: {len(test_loader)}")

        # --- Model Initialization ---
        input_size = config['model_params']['input_size']
        hidden_size = config['model_params']['hidden_size']
        num_classes = config['model_params']['num_classes']
        model = FeedForwardNN(input_size, hidden_size, num_classes).to(device)
        logger.info(f"Model initialized: {model}")

        # --- Optimizer and Loss Function ---
        learning_rate = config['training_params']['learning_rate']
        optimizer_name = config['training_params']['optimizer']
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            logger.error(f"Unsupported optimizer: {optimizer_name}. Only 'Adam' is supported.")
            raise TrainingError(
                f"Unsupported optimizer '{optimizer_name}'.",
                "Update 'optimizer' in config.yaml to 'Adam'."
            )
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Optimizer: {optimizer_name}, Learning Rate: {learning_rate}")
        logger.info(f"Loss Function: {criterion.__class__.__name__}")

        # --- Training Loop ---
        epochs = config['training_params']['epochs']
        checkpoint_dir = config['training_params']['checkpoint_dir']
        final_model_path = config['training_params']['final_model_path']
        best_accuracy = 0.0

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            logger.info(f"--- Epoch {epoch}/{epochs} ---")
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            val_loss, val_acc = validate_epoch(model, test_loader, criterion, device, epoch)



            is_best = val_acc > best_accuracy
            if is_best:
                best_accuracy = val_acc
                logger.info(f"New best validation accuracy: {best_accuracy:.4f}")

            save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_dir, best_accuracy, is_best)

        end_time = time.time()
        total_training_time = end_time - start_time
        logger.info(f"Training complete in {total_training_time:.2f} seconds.")
        logger.info(f"Best validation accuracy achieved: {best_accuracy:.4f}")

        # --- Final Model Save ---
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        torch.save(model.state_dict(), final_model_path)
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info("Training process finished successfully.")

    except (ConfigError, DataLoadingError, TrainingError) as e:
        logger.error(f"❌ Training failed: {e.message}", exc_info=True)
        print(f"\n❌ Training failed: {e.message}")
        print(f"💡 Fix: {e.fix}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled critical error occurred: {e}", exc_info=True)
        print(f"\n❌ A critical, unhandled error occurred during training.")
        print(f"📋 Error: {e}")
        print(f"💡 Fix: Review the logs in /workspace/logs/training.log for detailed context and traceback.")
        sys.exit(1)

if __name__ == "__main__":
    main()