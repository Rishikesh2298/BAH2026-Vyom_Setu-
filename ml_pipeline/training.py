"""
Training Pipeline for Climate Neural Network Models
"""

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class ClimateDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for climate data"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray, 
                 sequence_length: int = 30):
        """
        Initialize dataset
        
        Args:
            features: Climate features (time_steps, num_features)
            targets: Target variables (time_steps, num_targets)
            sequence_length: Sequence length for temporal models
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.features) - self.sequence_length
    
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length]
        return x, y

class ModelTrainer:
    """Training wrapper for climate models"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model
            device: Device to train on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"Initialized trainer on device: {device}")
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            
        Returns:
            Average epoch loss
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, (x, y) in enumerate(progress_bar):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader: DataLoader, 
                criterion: nn.Module) -> float:
        """
        Validate model
        
        Args:
            dataloader: Validation data loader
            criterion: Loss function
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Validating")
            for x, y in progress_bar:
                x = x.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(x)
                loss = criterion(outputs, y)
                
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int, learning_rate: float = 0.001,
             early_stopping_patience: int = 10) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader, criterion)
            self.history['val_loss'].append(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                patience_counter = 0
                self.save_model("best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        logger.info("Training completed")
        return self.history
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model loaded from {path}")

class EnsembleTrainer:
    """Train multiple models as ensemble"""
    
    def __init__(self, model_configs: list, device: str = 'cuda'):
        """
        Initialize ensemble trainer
        
        Args:
            model_configs: List of model configurations
            device: Device to train on
        """
        self.models = []
        self.trainers = []
        self.device = device
        
        logger.info(f"Initializing ensemble with {len(model_configs)} models")
        
        from ml_pipeline.climate_models import create_model
        
        for config in model_configs:
            model = create_model(config['type'], config['params'])
            trainer = ModelTrainer(model, device)
            
            self.models.append(model)
            self.trainers.append(trainer)
    
    def train_all(self, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int, learning_rate: float = 0.001):
        """
        Train all ensemble members
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        histories = []
        
        for i, trainer in enumerate(self.trainers):
            logger.info(f"\n{'='*50}")
            logger.info(f"Training ensemble member {i + 1}/{len(self.trainers)}")
            logger.info(f"{'='*50}")
            
            history = trainer.train(train_loader, val_loader, epochs, learning_rate)
            histories.append(history)
        
        return histories
    
    def save_ensemble(self, directory: str):
        """Save all ensemble members"""
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        for i, trainer in enumerate(self.trainers):
            path = f"{directory}/model_{i}.pt"
            trainer.save_model(path)
        
        logger.info(f"Ensemble saved to {directory}")

def generate_synthetic_data(n_samples: int = 10000, 
                           n_features: int = 100,
                           n_targets: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic climate data for testing
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_targets: Number of target variables
        
    Returns:
        Tuple of (features, targets)
    """
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")
    
    features = np.random.randn(n_samples, n_features).astype(np.float32)
    targets = np.random.randn(n_samples, n_targets).astype(np.float32)
    
    # Add some correlation
    targets = 0.3 * features[:, 0:n_targets] + 0.7 * targets
    
    return features, targets

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic data
    features, targets = generate_synthetic_data(n_samples=5000, n_features=50)
    
    # Create datasets
    train_size = int(0.7 * len(features))
    val_size = int(0.15 * len(features))
    
    train_data = ClimateDataset(features[:train_size], targets[:train_size])
    val_data = ClimateDataset(features[train_size:train_size+val_size], 
                             targets[train_size:train_size+val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Train LSTM model
    from ml_pipeline.climate_models import LSTMClimateForecaster
    
    model = LSTMClimateForecaster(input_size=50, hidden_size=128, output_size=1)
    trainer = ModelTrainer(model)
    
    history = trainer.train(train_loader, val_loader, epochs=50, 
                           learning_rate=0.001, early_stopping_patience=5)
    
    logger.info("Training completed successfully!")
