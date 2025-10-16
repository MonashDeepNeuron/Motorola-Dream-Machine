#!/usr/bin/env python3
"""
Simple model training script for EEG2Arm
This creates a basic training pipeline for the AI model.
"""

import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model.eeg_model import EEG2Arm

# Simple ring topology for electrodes
def create_electrode_edges(n_elec: int):
    edges = [(i, (i + 1) % n_elec) for i in range(n_elec)]
    for i in range(n_elec // 2):
        edges.append((i, i + n_elec // 2))
    return edges


class DummyEEGDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.
    Replace with real EEG data loading.
    """
    
    def __init__(self, n_samples=1000, n_elec=32, n_bands=5, n_frames=12, n_classes=5):
        self.n_samples = n_samples
        self.n_elec = n_elec
        self.n_bands = n_bands
        self.n_frames = n_frames
        self.n_classes = n_classes
        
        # Generate dummy data
        self.data = torch.randn(n_samples, n_elec, n_bands, n_frames)
        self.labels = torch.randint(0, n_classes, (n_samples,))
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}', end='\r')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train EEG2Arm model')
    parser.add_argument('--n-elec', type=int, default=32, help='Number of electrodes')
    parser.add_argument('--n-bands', type=int, default=5, help='Number of frequency bands')
    parser.add_argument('--n-frames', type=int, default=12, help='Number of time frames')
    parser.add_argument('--n-classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--output-dir', default='checkpoints', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("EEG2Arm Model Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Electrodes: {args.n_elec}")
    print(f"  Frequency bands: {args.n_bands}")
    print(f"  Time frames: {args.n_frames}")
    print(f"  Classes: {args.n_classes}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    
    # Create model
    print("\nInitializing model...")
    edges = create_electrode_edges(args.n_elec)
    model = EEG2Arm(
        n_elec=args.n_elec,
        n_bands=args.n_bands,
        clip_length=None,
        n_classes=args.n_classes,
        pointwise_groups=1,
        edges=edges
    ).to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create datasets
    print("\nCreating datasets...")
    print("⚠️  WARNING: Using dummy data for demonstration")
    print("   Replace DummyEEGDataset with real data loader")
    
    train_dataset = DummyEEGDataset(
        n_samples=1000,
        n_elec=args.n_elec,
        n_bands=args.n_bands,
        n_frames=args.n_frames,
        n_classes=args.n_classes
    )
    
    val_dataset = DummyEEGDataset(
        n_samples=200,
        n_elec=args.n_elec,
        n_bands=args.n_bands,
        n_frames=args.n_frames,
        n_classes=args.n_classes
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    print("\nStarting training...")
    best_val_acc = 0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'n_elec': args.n_elec,
                    'n_bands': args.n_bands,
                    'n_frames': args.n_frames,
                    'n_classes': args.n_classes
                }
            }, checkpoint_path)
            print(f"  ✅ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Save final model
    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': {
            'n_elec': args.n_elec,
            'n_bands': args.n_bands,
            'n_frames': args.n_frames,
            'n_classes': args.n_classes
        }
    }, final_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Best model saved to: {checkpoint_path}")
    print(f"  Final model saved to: {final_path}")
    print(f"  Training history: {history_path}")
    print("=" * 60)
    
    print("\nTo use the trained model:")
    print(f"  python eeg_pipeline/ai_consumer/ai_consumer.py --model-path {checkpoint_path}")


if __name__ == '__main__':
    main()
