#!/usr/bin/env python3
"""
Demo Script for EEG-to-Robot Pipeline
=====================================

This demo shows how the pipeline works, even without all dependencies installed.
It simulates the complete flow from EEG data to robot commands.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

def demo_eeg_processing():
    """Demonstrate EEG signal processing"""
    print("=== EEG Signal Processing Demo ===")
    
    # Simulate EEG data (32 channels, 1024 samples, 256 Hz = 4 seconds)
    n_channels = 32
    n_samples = 1024
    sample_rate = 256
    
    print(f"Simulating EEG data: {n_channels} channels, {n_samples} samples at {sample_rate} Hz")
    
    # Generate realistic EEG-like signals
    time_axis = np.linspace(0, n_samples/sample_rate, n_samples)
    eeg_data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Mix of different frequency components
        alpha = np.sin(2 * np.pi * 10 * time_axis) * np.random.uniform(0.5, 1.5)  # 10 Hz alpha
        beta = np.sin(2 * np.pi * 20 * time_axis) * np.random.uniform(0.2, 0.8)   # 20 Hz beta  
        noise = np.random.normal(0, 0.1, n_samples)                               # Background noise
        
        eeg_data[ch] = alpha + beta + noise
    
    # Scale to realistic EEG voltage range (microvolts)
    eeg_data = eeg_data * 1e-5  # Convert to volts (typical EEG range)
    
    print(f"Generated EEG data shape: {eeg_data.shape}")
    print(f"Voltage range: {eeg_data.min():.2e} to {eeg_data.max():.2e} V")
    
    return eeg_data, time_axis

def demo_frequency_analysis(eeg_data, sample_rate=256):
    """Demonstrate frequency domain analysis"""
    print("\n=== Frequency Domain Analysis Demo ===")
    
    # Define frequency bands
    freq_bands = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 30),
        "gamma": (30, 45)
    }
    
    n_channels, n_samples = eeg_data.shape
    window_size = 4.0  # seconds
    window_samples = int(window_size * sample_rate)
    
    print(f"Analyzing {len(freq_bands)} frequency bands")
    print(f"Window size: {window_size} seconds ({window_samples} samples)")
    
    # Compute power spectral density for each channel and band
    band_powers = np.zeros((n_channels, len(freq_bands)))
    
    for ch in range(n_channels):
        # Use the full signal for simplicity
        signal = eeg_data[ch]
        
        # Compute FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Compute power spectral density
        psd = np.abs(fft_vals) ** 2
        
        # Extract power in each frequency band
        for i, (band_name, (fmin, fmax)) in enumerate(freq_bands.items()):
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.mean(psd[freq_mask])
            band_powers[ch, i] = band_power
    
    print(f"Extracted band power features shape: {band_powers.shape}")
    
    # Show average power across channels for each band
    avg_powers = np.mean(band_powers, axis=0)
    print("\nAverage power by frequency band:")
    for i, band_name in enumerate(freq_bands.keys()):
        print(f"  {band_name}: {avg_powers[i]:.2e}")
    
    return band_powers

def demo_model_prediction(features):
    """Demonstrate model prediction (without actual PyTorch)"""
    print("\n=== Model Prediction Demo ===")
    
    n_channels, n_bands = features.shape
    n_classes = 5
    
    print(f"Input features: {n_channels} channels × {n_bands} frequency bands")
    print(f"Model classes: {n_classes} (rest, left, right, forward, backward)")
    
    # Simulate model processing steps
    print("\nModel architecture simulation:")
    print("1. CNN: Spatial-temporal feature extraction")
    
    # Simulate CNN output
    cnn_features = np.mean(features, axis=0)  # Simplified spatial pooling
    print(f"   CNN output shape: {cnn_features.shape}")
    
    print("2. GCN: Graph convolution over electrodes")
    # Simulate graph convolution
    gcn_features = np.concatenate([cnn_features, cnn_features * 0.5])  # Simulate feature expansion
    print(f"   GCN output shape: {gcn_features.shape}")
    
    print("3. Transformer: Temporal sequence modeling")
    # Simulate transformer output
    transformer_features = np.mean(gcn_features) * np.ones(32)  # Simulate attention pooling
    print(f"   Transformer output shape: {transformer_features.shape}")
    
    print("4. Classification head: Final prediction")
    # Simulate classification with simple linear transformation
    class_logits = np.random.normal(0, 1, n_classes)
    
    # Add some bias based on input features to make it more realistic
    avg_power = np.mean(features)
    if avg_power > np.median(features.flatten()):
        class_logits[1:3] += 0.5  # Bias toward left/right movement
    else:
        class_logits[0] += 0.5    # Bias toward rest
    
    # Apply softmax
    exp_logits = np.exp(class_logits - np.max(class_logits))
    probabilities = exp_logits / np.sum(exp_logits)
    predicted_class = np.argmax(probabilities)
    
    class_names = ["rest", "left", "right", "forward", "backward"]
    
    print(f"   Raw logits: {class_logits}")
    print(f"   Probabilities: {probabilities}")
    print(f"   Predicted class: {predicted_class} ({class_names[predicted_class]})")
    print(f"   Confidence: {probabilities[predicted_class]:.2%}")
    
    return predicted_class, probabilities

def demo_robot_control(predicted_class):
    """Demonstrate robot command generation"""
    print("\n=== Robot Control Demo ===")
    
    # Load or create command mapping
    command_mapping = {
        0: {"dx": 0.0, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0},  # Rest
        1: {"dx": 0.05, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Left
        2: {"dx": -0.05, "dy": 0.0, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Right
        3: {"dx": 0.0, "dy": 0.05, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Forward
        4: {"dx": 0.0, "dy": -0.05, "dz": 0.0, "drx": 0.0, "dry": 0.0, "drz": 0.0}, # Backward
    }
    
    command = command_mapping[predicted_class]
    class_names = ["rest", "left", "right", "forward", "backward"]
    
    print(f"Predicted class: {predicted_class} ({class_names[predicted_class]})")
    print(f"Robot command: {command}")
    
    # Write command to file (like the real system)
    command_file = Path("ursim_test_v1/asynchronous_deltas.jsonl")
    command_file.parent.mkdir(exist_ok=True)
    
    with open(command_file, 'a') as f:
        json.dump(command, f)
        f.write('\n')
    
    print(f"Command written to: {command_file}")
    
    return command

def demo_full_pipeline():
    """Run complete pipeline demonstration"""
    print("🧠 EEG-to-Robot Pipeline Demo")
    print("=" * 50)
    print("This demo simulates the complete pipeline without requiring")
    print("all dependencies to be installed.")
    print()
    
    # Step 1: EEG Data Processing
    eeg_data, time_axis = demo_eeg_processing()
    
    # Step 2: Frequency Analysis
    band_powers = demo_frequency_analysis(eeg_data)
    
    # Step 3: Model Prediction
    predicted_class, probabilities = demo_model_prediction(band_powers)
    
    # Step 4: Robot Control
    command = demo_robot_control(predicted_class)
    
    print("\n=== Pipeline Summary ===")
    print(f"✅ Processed EEG data: {eeg_data.shape[0]} channels")
    print(f"✅ Extracted features: {band_powers.shape[1]} frequency bands")
    print(f"✅ Model prediction: Class {predicted_class}")
    print(f"✅ Robot command: {command}")
    print()
    print("🎉 Pipeline demo completed successfully!")
    
    return {
        'eeg_data': eeg_data,
        'features': band_powers,
        'prediction': predicted_class,
        'command': command
    }

def demo_real_time_simulation(duration=30):
    """Simulate real-time processing"""
    print(f"\n🔄 Real-time Processing Simulation ({duration} seconds)")
    print("=" * 50)
    
    command_file = Path("ursim_test_v1/asynchronous_deltas.jsonl")
    command_file.parent.mkdir(exist_ok=True)
    
    # Clear previous commands
    with open(command_file, 'w') as f:
        pass
    
    print("Processing EEG windows every 2 seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        for i in range(duration // 2):
            print(f"\nWindow {i+1}:")
            
            # Simulate EEG processing
            eeg_data, _ = demo_eeg_processing()
            features = demo_frequency_analysis(eeg_data)
            predicted_class, _ = demo_model_prediction(features)
            command = demo_robot_control(predicted_class)
            
            print(f"  → Command sent: {command}")
            
            # Wait for next window
            time.sleep(2)
        
        print(f"\n✅ Real-time simulation completed!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Simulation stopped by user")
    
    # Show command history
    if command_file.exists():
        with open(command_file, 'r') as f:
            commands = [json.loads(line) for line in f if line.strip()]
        
        print(f"\n📋 Generated {len(commands)} robot commands")
        print("Recent commands:")
        for cmd in commands[-5:]:
            print(f"  {cmd}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EEG Pipeline Demo")
    parser.add_argument('--mode', choices=['single', 'realtime'], default='single',
                       help='Demo mode: single run or real-time simulation')
    parser.add_argument('--duration', type=int, default=30,
                       help='Duration for real-time simulation (seconds)')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        results = demo_full_pipeline()
    elif args.mode == 'realtime':
        demo_real_time_simulation(args.duration)
    
    print("\n" + "="*50)
    print("Demo completed! To run the actual pipeline:")
    print("1. Run ./setup_pipeline.sh to install dependencies")
    print("2. Run python prepare_data.py --edf-files eeg_files/*.edf")
    print("3. Run python train_model.py")
    print("4. Run python run_inference.py")
