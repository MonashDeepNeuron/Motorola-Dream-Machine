#!/usr/bin/env python3
"""
EEG Inference and Robot Control Script
=====================================

This script loads a trained model and processes EEG data to generate robot commands.
"""

import os
import sys
import json
import time
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add project paths
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
sys.path.insert(0, project_root)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available. Please run setup_pipeline.sh first.")
    TORCH_AVAILABLE = False

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    print("MNE not available. Please run setup_pipeline.sh first.")
    MNE_AVAILABLE = False

if TORCH_AVAILABLE:
    from model.eeg_model import EEG2Arm

class RobotController:
    """Handles robot command generation and output"""
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.command_file = Path(self.config["paths"]["robot_command_file"])
        self.command_mapping = {int(k): v for k, v in self.config["robot_commands"].items()}
        
        # Ensure command file exists
        self.command_file.parent.mkdir(exist_ok=True)
        if not self.command_file.exists():
            self.command_file.touch()
    
    def send_command(self, prediction: int):
        """Send robot command based on prediction"""
        command = self.command_mapping.get(prediction, self.command_mapping[0])
        
        # Append to JSONL file
        with open(self.command_file, 'a') as f:
            json.dump(command, f)
            f.write('\n')
        
        print(f"Robot command: Class {prediction} -> {command}")
        return command
    
    def clear_commands(self):
        """Clear the command file"""
        with open(self.command_file, 'w') as f:
            pass
        print("Cleared robot command file")

class EEGInference:
    """Handles EEG inference"""
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model parameters
        self.n_electrodes = self.config["pipeline_config"]["n_electrodes"]
        self.n_bands = self.config["pipeline_config"]["n_frequency_bands"]
        self.n_classes = self.config["pipeline_config"]["n_classes"]
        
        # Initialize model
        self.model = EEG2Arm(
            n_elec=self.n_electrodes,
            n_bands=self.n_bands,
            n_classes=self.n_classes,
            cnn_time_pool=self.config["model_config"]["cnn_time_pool"],
            pointwise_groups=self.config["model_config"]["pointwise_groups"]
        ).to(self.device)
        
        self.model.eval()
        
    def load_model_weights(self, model_path: str):
        """Load trained model weights"""
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded model weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}")
            print("Using random weights - train the model first!")
    
    def predict(self, features: np.ndarray, sequence_length: int = 10) -> int:
        """Make prediction from features"""
        # Convert to tensor and add sequence dimension
        if features.ndim == 2:  # (n_channels, n_bands)
            features = np.expand_dims(features, axis=-1)  # (n_channels, n_bands, 1)
            features = np.repeat(features, sequence_length, axis=-1)  # (n_channels, n_bands, seq_len)
        
        # Add batch dimension and convert to tensor
        input_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction

def process_prepared_data(data_dir: str = "data/processed", model_path: str = None):
    """Process already prepared data for inference"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available for inference")
        return
    
    data_path = Path(data_dir)
    if not (data_path / "features.npy").exists():
        print(f"No processed data found in {data_dir}")
        print("Please run prepare_data.py first")
        return
    
    # Load data
    features = np.load(data_path / "features.npy")
    labels = np.load(data_path / "labels.npy")
    
    print(f"Loaded {len(features)} feature windows")
    
    # Initialize inference and robot controller
    inference = EEGInference()
    robot_controller = RobotController()
    
    # Load model if provided
    if model_path:
        inference.load_model_weights(model_path)
    else:
        model_path = inference.config["paths"]["model_weights"]
        inference.load_model_weights(model_path)
    
    # Clear previous commands
    robot_controller.clear_commands()
    
    # Process features and generate commands
    correct_predictions = 0
    
    print("\nProcessing features and generating robot commands...")
    print("Press Ctrl+C to stop")
    
    try:
        for i, (feature, true_label) in enumerate(zip(features, labels)):
            # Make prediction
            predicted_class = inference.predict(feature)
            
            # Send robot command
            robot_controller.send_command(predicted_class)
            
            # Track accuracy
            if predicted_class == true_label:
                correct_predictions += 1
            
            # Print progress
            if i % 10 == 0:
                accuracy = 100 * correct_predictions / (i + 1)
                print(f"Processed {i+1}/{len(features)} windows, "
                      f"Accuracy: {accuracy:.1f}%, "
                      f"Predicted: {predicted_class}, True: {true_label}")
            
            # Wait between commands
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    final_accuracy = 100 * correct_predictions / len(features)
    print(f"\nFinal accuracy: {final_accuracy:.1f}%")

def simulate_real_time_demo():
    """Simulate real-time processing with dummy data"""
    if not TORCH_AVAILABLE:
        print("PyTorch not available for simulation")
        return
    
    print("Simulating real-time EEG processing...")
    
    # Initialize inference and robot controller
    inference = EEGInference()
    robot_controller = RobotController()
    
    # Load model
    model_path = inference.config["paths"]["model_weights"]
    inference.load_model_weights(model_path)
    
    # Clear previous commands
    robot_controller.clear_commands()
    
    print("Starting simulation (Press Ctrl+C to stop)...")
    
    try:
        window_count = 0
        while True:
            # Generate dummy EEG features (replace with real streaming data)
            n_channels = inference.n_electrodes
            n_bands = inference.n_bands
            
            # Simulate band power features
            dummy_features = np.random.exponential(scale=1e-6, size=(n_channels, n_bands))
            
            # Add some patterns for different "mental states"
            time_factor = window_count % 100
            if time_factor < 20:  # "Rest" state
                dummy_features *= 0.5
            elif time_factor < 40:  # "Left" state
                dummy_features[:n_channels//2, 1:3] *= 2  # Increase alpha/beta on left
            elif time_factor < 60:  # "Right" state
                dummy_features[n_channels//2:, 1:3] *= 2  # Increase alpha/beta on right
            elif time_factor < 80:  # "Forward" state
                dummy_features[:, 3:] *= 1.5  # Increase beta/gamma
            else:  # "Backward" state
                dummy_features[:, :2] *= 1.5  # Increase delta/theta
            
            # Make prediction
            predicted_class = inference.predict(dummy_features)
            
            # Send robot command
            robot_controller.send_command(predicted_class)
            
            window_count += 1
            
            # Print status
            if window_count % 10 == 0:
                print(f"Processed {window_count} windows")
            
            # Wait for next window (simulate real-time rate)
            time.sleep(0.5)  # 2 Hz processing rate
    
    except KeyboardInterrupt:
        print(f"\nSimulation stopped after {window_count} windows")

def main():
    parser = argparse.ArgumentParser(description="EEG Inference and Robot Control")
    parser.add_argument('--mode', choices=['data', 'simulate'], default='data',
                       help='Processing mode: "data" for prepared data, "simulate" for demo')
    parser.add_argument('--data-dir', default='data/processed',
                       help='Directory containing processed data')
    parser.add_argument('--model-path', 
                       help='Path to trained model weights')
    parser.add_argument('--config', default='pipeline_config.json',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    if args.mode == 'data':
        process_prepared_data(args.data_dir, args.model_path)
    elif args.mode == 'simulate':
        simulate_real_time_demo()

if __name__ == "__main__":
    main()
