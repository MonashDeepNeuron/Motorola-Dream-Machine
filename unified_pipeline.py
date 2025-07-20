#!/usr/bin/env python3
"""
Unified EEG-to-Robot Pipeline
=============================

This script creates a complete pipeline that:
1. Processes EEG files (training or real-time streaming)
2. Extracts frequency domain features (FFT-based band power)
3. Feeds data through the EEG model for robot control predictions
4. Outputs commands to the UR sim asynchronous system

No Kafka required - everything runs on a single system.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
import threading
import queue

# Add project paths to sys.path
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "eeg_pipeline"))

# Import EEG pipeline components
import mne
from eeg_pipeline.analysis.bands import compute_window_band_power, DefaultBands
from eeg_pipeline.schemas.eeg_schemas import EEGBatch, WindowBandPower
from model.eeg_model import EEG2Arm

# Configuration
class PipelineConfig:
    # EEG Processing
    SAMPLE_RATE = 256  # Hz
    WINDOW_SIZE = 4.0  # seconds
    STEP_SIZE = 2.0    # seconds
    BATCH_SIZE = 256   # samples per batch
    
    # Model
    N_ELECTRODES = 32  # Adjust based on your headset
    N_FREQUENCY_BANDS = 5  # delta, theta, alpha, beta, gamma
    N_CLASSES = 5      # Robot control commands
    
    # Robot Control
    ROBOT_COMMAND_FILE = "ursim_test_v1/asynchronous_deltas.jsonl"
    COMMAND_RATE = 10  # Hz - how often to send robot commands
    
    # Band definitions
    FREQ_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8), 
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45)
    }

class EEGProcessor:
    """Handles EEG data processing and feature extraction"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.window_buffer = deque(maxlen=int(config.WINDOW_SIZE * config.SAMPLE_RATE))
        
    def process_edf_file(self, edf_path: str) -> Tuple[np.ndarray, List[str], float]:
        """Load and preprocess EDF file"""
        print(f"Loading EDF file: {edf_path}")
        
        # Load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
        raw.rename_channels(lambda x: x.strip('. ').replace('..', '').upper())
        
        # Get data in volts
        data_volts = raw.get_data(units='V')
        sfreq = raw.info['sfreq']
        channel_names = raw.ch_names
        
        print(f"Loaded: {data_volts.shape[1]} samples, {len(channel_names)} channels, {sfreq} Hz")
        
        # Resample if needed
        if sfreq != self.config.SAMPLE_RATE:
            print(f"Resampling from {sfreq} Hz to {self.config.SAMPLE_RATE} Hz")
            raw.resample(self.config.SAMPLE_RATE)
            data_volts = raw.get_data(units='V')
            sfreq = self.config.SAMPLE_RATE
            
        return data_volts, channel_names, sfreq
    
    def extract_band_power_features(self, data_window: np.ndarray, sfreq: float) -> np.ndarray:
        """Extract frequency domain features using FFT"""
        n_channels, n_samples = data_window.shape
        
        # Create MNE Raw object for band power computation
        info = mne.create_info(ch_names=[f"CH{i}" for i in range(n_channels)], 
                              sfreq=sfreq, ch_types='eeg')
        raw_window = mne.io.RawArray(data_window, info)
        
        # Compute band power
        window_results, _ = compute_window_band_power(
            raw_window, 
            window_size_s=self.config.WINDOW_SIZE,
            step_size_s=self.config.WINDOW_SIZE,  # No overlap for single window
            bands=self.config.FREQ_BANDS
        )
        
        # Organize features: [n_channels, n_bands]
        features = np.zeros((n_channels, len(self.config.FREQ_BANDS)))
        
        for result in window_results:
            ch_idx = int(result.channel_label.replace("CH", ""))
            band_names = list(self.config.FREQ_BANDS.keys())
            band_idx = band_names.index(result.band_name)
            features[ch_idx, band_idx] = result.power
            
        return features
    
    def create_model_input(self, band_power_features: np.ndarray, sequence_length: int = 10) -> torch.Tensor:
        """Convert band power features to model input format"""
        # Model expects: (batch, n_electrodes, n_bands, time_steps)
        n_channels, n_bands = band_power_features.shape
        
        # For real-time processing, we maintain a buffer of recent features
        # For simplicity, we'll repeat the current window
        features_expanded = np.expand_dims(band_power_features, axis=-1)
        features_sequence = np.repeat(features_expanded, sequence_length, axis=-1)
        
        # Add batch dimension and convert to tensor
        model_input = torch.from_numpy(features_sequence).float().unsqueeze(0)
        
        return model_input

class RobotController:
    """Handles robot command generation and output"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.command_file = Path(config.ROBOT_COMMAND_FILE)
        self.command_mapping = {
            0: {"dx": 0.0, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0},  # Rest
            1: {"dx": 0.05, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Move right
            2: {"dx": -0.05, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Move left
            3: {"dx": 0.0, "dy": 0.05, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Move forward
            4: {"dx": 0.0, "dy": -0.05, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Move backward
        }
        
        # Ensure command file exists
        self.command_file.parent.mkdir(exist_ok=True)
        if not self.command_file.exists():
            self.command_file.touch()
    
    def send_command(self, prediction: torch.Tensor):
        """Convert model prediction to robot command and append to file"""
        # Get predicted class
        predicted_class = torch.argmax(prediction, dim=1).item()
        
        # Get command
        command = self.command_mapping.get(predicted_class, self.command_mapping[0])
        
        # Append to JSONL file
        with open(self.command_file, 'a') as f:
            json.dump(command, f)
            f.write('\n')
            
        print(f"Robot command: Class {predicted_class} -> {command}")
        
        return command

class UnifiedPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.eeg_processor = EEGProcessor(config)
        self.robot_controller = RobotController(config)
        
        # Initialize model
        self.model = EEG2Arm(
            n_elec=config.N_ELECTRODES,
            n_bands=config.N_FREQUENCY_BANDS,
            n_classes=config.N_CLASSES,
            clip_length=None  # Variable length
        )
        
        # Set model to evaluation mode (for inference)
        self.model.eval()
        
        print("Unified pipeline initialized")
    
    def load_model_weights(self, model_path: str):
        """Load pre-trained model weights"""
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Model weights not found at {model_path}, using random initialization")
    
    def train_mode(self, edf_files: List[str], epochs: int = 10):
        """Training mode - train the model on EDF files"""
        print(f"Training mode: Processing {len(edf_files)} files")
        
        # Prepare training data
        all_features = []
        all_labels = []
        
        for edf_file in edf_files:
            data_volts, channel_names, sfreq = self.eeg_processor.process_edf_file(edf_file)
            
            # Extract features from sliding windows
            window_samples = int(self.config.WINDOW_SIZE * sfreq)
            step_samples = int(self.config.STEP_SIZE * sfreq)
            
            for start_idx in range(0, data_volts.shape[1] - window_samples + 1, step_samples):
                window_data = data_volts[:, start_idx:start_idx + window_samples]
                features = self.eeg_processor.extract_band_power_features(window_data, sfreq)
                
                # For training, we need labels - this is a simplified example
                # In practice, you'd get labels from annotations or other sources
                label = np.random.randint(0, self.config.N_CLASSES)  # Random label for demo
                
                all_features.append(features)
                all_labels.append(label)
        
        # Convert to tensors
        X = torch.stack([self.eeg_processor.create_model_input(f).squeeze(0) for f in all_features])
        y = torch.tensor(all_labels, dtype=torch.long)
        
        print(f"Training data shape: {X.shape}, Labels: {y.shape}")
        
        # Simple training loop
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        
        # Save model
        model_save_path = "model/trained_eeg_model.pth"
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    
    def inference_mode(self, edf_file: str):
        """Inference mode - process EDF file and generate robot commands"""
        print(f"Inference mode: Processing {edf_file}")
        
        # Load data
        data_volts, channel_names, sfreq = self.eeg_processor.process_edf_file(edf_file)
        
        # Process in sliding windows
        window_samples = int(self.config.WINDOW_SIZE * sfreq)
        step_samples = int(self.config.STEP_SIZE * sfreq)
        
        self.model.eval()
        with torch.no_grad():
            for start_idx in range(0, data_volts.shape[1] - window_samples + 1, step_samples):
                window_data = data_volts[:, start_idx:start_idx + window_samples]
                
                # Extract features
                features = self.eeg_processor.extract_band_power_features(window_data, sfreq)
                
                # Create model input
                model_input = self.eeg_processor.create_model_input(features)
                
                # Get prediction
                prediction = self.model(model_input)
                
                # Send robot command
                command = self.robot_controller.send_command(prediction)
                
                # Wait before next command
                time.sleep(1.0 / self.config.COMMAND_RATE)
    
    def real_time_mode(self, device_interface=None):
        """Real-time mode - process streaming EEG data"""
        print("Real-time mode: Streaming EEG processing")
        print("Note: Real-time streaming requires Emotiv SDK integration")
        
        # This is a placeholder for real-time streaming
        # In practice, you'd integrate with Emotiv SDK or other streaming interface
        
        buffer = deque(maxlen=int(self.config.WINDOW_SIZE * self.config.SAMPLE_RATE))
        
        # Simulate streaming data (replace with actual streaming code)
        dummy_data = np.random.randn(self.config.N_ELECTRODES, 1000) * 1e-5  # Simulate EEG in volts
        
        self.model.eval()
        with torch.no_grad():
            for sample_idx in range(dummy_data.shape[1]):
                # Add new sample to buffer
                buffer.append(dummy_data[:, sample_idx])
                
                # Process when buffer is full
                if len(buffer) == buffer.maxlen:
                    window_data = np.array(buffer).T  # Shape: (n_channels, window_samples)
                    
                    # Extract features
                    features = self.eeg_processor.extract_band_power_features(window_data, self.config.SAMPLE_RATE)
                    
                    # Create model input
                    model_input = self.eeg_processor.create_model_input(features)
                    
                    # Get prediction
                    prediction = self.model(model_input)
                    
                    # Send robot command
                    command = self.robot_controller.send_command(prediction)
                    
                    # Clear some of the buffer for sliding window
                    for _ in range(int(self.config.STEP_SIZE * self.config.SAMPLE_RATE)):
                        if buffer:
                            buffer.popleft()
                
                # Simulate real-time delay
                time.sleep(1.0 / self.config.SAMPLE_RATE)

def main():
    parser = argparse.ArgumentParser(description="Unified EEG-to-Robot Pipeline")
    parser.add_argument('--mode', choices=['train', 'inference', 'realtime'], required=True,
                       help='Pipeline mode')
    parser.add_argument('--edf-files', nargs='+', 
                       help='EDF files for training/inference mode')
    parser.add_argument('--model-path', default='model/trained_eeg_model.pth',
                       help='Path to load/save model weights')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--config-file', 
                       help='JSON config file to override defaults')
    
    args = parser.parse_args()
    
    # Load configuration
    config = PipelineConfig()
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Initialize pipeline
    pipeline = UnifiedPipeline(config)
    
    # Load model if available
    if args.mode in ['inference', 'realtime']:
        pipeline.load_model_weights(args.model_path)
    
    # Run appropriate mode
    if args.mode == 'train':
        if not args.edf_files:
            print("Error: --edf-files required for training mode")
            return
        pipeline.train_mode(args.edf_files, args.epochs)
        
    elif args.mode == 'inference':
        if not args.edf_files:
            print("Error: --edf-files required for inference mode")
            return
        for edf_file in args.edf_files:
            pipeline.inference_mode(edf_file)
            
    elif args.mode == 'realtime':
        pipeline.real_time_mode()

if __name__ == "__main__":
    main()
