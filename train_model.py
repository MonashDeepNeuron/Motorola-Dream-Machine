#!/usr/bin/env python3
"""
EEG Model Training Script
========================

Train the EEG-to-Robot model using preprocessed data.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import argparse
from datetime import datetime

# Add project paths
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
sys.path.insert(0, project_root)

from model.eeg_model import EEG2Arm

class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence_length: int = 10):
        """
        Args:
            features: (n_windows, n_channels, n_bands)
            labels: (n_windows,)
            sequence_length: Number of time steps for model input
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        # Get feature window
        feature = self.features[idx]  # (n_channels, n_bands)
        label = self.labels[idx]
        
        # Expand to sequence format: (n_channels, n_bands, sequence_length)
        # For simplicity, repeat the same window
        feature_seq = feature.unsqueeze(-1).repeat(1, 1, self.sequence_length)
        
        return feature_seq, label

class EEGTrainer:
    """Training manager for EEG model"""
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model parameters
        self.n_electrodes = self.config["pipeline_config"]["n_electrodes"]
        self.n_bands = self.config["pipeline_config"]["n_frequency_bands"]
        self.n_classes = self.config["pipeline_config"]["n_classes"]
        
        # Training parameters
        self.learning_rate = self.config["model_config"]["learning_rate"]
        self.batch_size = 32
        self.epochs = 50
        
        # Initialize model
        self.model = EEG2Arm(
            n_elec=self.n_electrodes,
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            cnn_time_pool=self.config["model_config"]["cnn_time_pool"],
            pointwise_groups=self.config["model_config"]["pointwise_groups"]
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        
    def load_data(self, data_dir: str = "data/processed"):
        """Load preprocessed data"""
        data_path = Path(data_dir)
        
        features = np.load(data_path / "features.npy")
        labels = np.load(data_path / "labels.npy")
        
        with open(data_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded data: {features.shape[0]} samples")
        print(f"Feature shape: {features.shape}")
        print(f"Label distribution: {metadata['label_distribution']}")
        
        # Adjust model if needed based on actual data dimensions
        actual_channels = features.shape[1]
        if actual_channels != self.n_electrodes:
            print(f"Adjusting model: {self.n_electrodes} -> {actual_channels} electrodes")
            self.n_electrodes = actual_channels
            self.model = EEG2Arm(
                n_elec=self.n_electrodes,
                n_bands=self.n_bands,
                n_classes=self.n_classes,
                cnn_time_pool=self.config["model_config"]["cnn_time_pool"],
                pointwise_groups=self.config["model_config"]["pointwise_groups"]
            ).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        return features, labels, metadata
    
    def create_data_loaders(self, features: np.ndarray, labels: np.ndarray, 
                           train_split: float = 0.8, sequence_length: int = 10):
        """Create train/validation data loaders"""
        
        # Create dataset
        dataset = EEGDataset(features, labels, sequence_length)
        
        # Split dataset
        train_size = int(train_split * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                              shuffle=False, num_workers=2)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}, '
                      f'Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = test_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, data_dir: str = "data/processed", epochs: int = None):
        """Full training loop"""
        if epochs is not None:
            self.epochs = epochs
        
        # Load data
        features, labels, metadata = self.load_data(data_dir)
        train_loader, val_loader = self.create_data_loaders(features, labels)
        
        # Training history
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_acc = 0
        best_model_path = None
        
        print(f"\nStarting training for {self.epochs} epochs...")
        print("="*50)
        
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            print("-" * 30)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = self.save_model(epoch, val_acc)
                print(f"New best model saved: {best_model_path}")
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best model: {best_model_path}")
        
        # Save training history
        self.save_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        return best_model_path
    
    def save_model(self, epoch: int, val_acc: float):
        """Save model checkpoint"""
        model_dir = Path("model/checkpoints")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eeg_model_epoch{epoch+1}_acc{val_acc:.1f}_{timestamp}.pth"
        filepath = model_dir / filename
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'config': self.config
        }, filepath)
        
        # Also save as latest
        latest_path = Path(self.config["paths"]["model_weights"])
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), latest_path)
        
        return str(filepath)
    
    def save_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """Save training metrics"""
        history = {
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        log_dir = Path(self.config["paths"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        with open(log_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Train EEG Model")
    parser.add_argument('--data-dir', default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--config', default='pipeline_config.json',
                       help='Configuration file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Check if data exists
    data_path = Path(args.data_dir)
    if not (data_path / "features.npy").exists():
        print(f"No processed data found in {args.data_dir}")
        print("Please run prepare_data.py first to preprocess EDF files")
        return
    
    # Initialize trainer
    trainer = EEGTrainer(args.config)
    trainer.batch_size = args.batch_size
    
    # Train model
    best_model = trainer.train(args.data_dir, args.epochs)
    print(f"\nTraining complete! Best model saved to: {best_model}")

if __name__ == "__main__":
    main()
