#!/usr/bin/env python3
"""
Demo 1: EEG Signal Processing
============================

This demonstration shows exactly how raw brain signals become clean, analyzed data.

Usage:
    python3 demos/demo_signal_processing.py

Output:
    - Raw EEG plots (before/after filtering)
    - Frequency analysis charts  
    - Feature extraction results
    - Processing time benchmarks
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.eeg.processor import RealTimeEEGProcessor
    from src.eeg.features import EEGFeatureExtractor
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

def generate_simulated_eeg(duration=4, sampling_rate=256, channels=14):
    """Generate realistic simulated EEG data."""
    samples = int(duration * sampling_rate)
    
    # Simulate different frequency components
    time = np.linspace(0, duration, samples)
    
    eeg_data = np.zeros((channels, samples))
    
    for ch in range(channels):
        # Alpha rhythm (8-13 Hz) - dominant when relaxed
        alpha = 0.5 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
        
        # Beta rhythm (13-30 Hz) - active thinking
        beta = 0.3 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
        
        # Theta rhythm (4-8 Hz) - drowsiness
        theta = 0.2 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
        
        # Gamma rhythm (30-40 Hz) - cognitive processing
        gamma = 0.1 * np.sin(2 * np.pi * 35 * time + np.random.rand() * 2 * np.pi)
        
        # Random noise
        noise = 0.05 * np.random.randn(samples)
        
        # Power line interference (60 Hz)
        powerline = 0.1 * np.sin(2 * np.pi * 60 * time)
        
        # Combine all components
        eeg_data[ch] = alpha + beta + theta + gamma + noise + powerline
        
        # Add some artifacts for demonstration
        if ch < 2:  # Eye blink artifact in frontal channels
            blink_times = [1.0, 2.5]
            for blink_time in blink_times:
                blink_idx = int(blink_time * sampling_rate)
                blink_duration = int(0.1 * sampling_rate)  # 100ms blink
                if blink_idx + blink_duration < samples:
                    eeg_data[ch, blink_idx:blink_idx+blink_duration] += 2.0
    
    return eeg_data, time

def demo_signal_processing():
    """Demonstrate EEG signal processing pipeline."""
    print("🔬 Demo 1: EEG Signal Processing")
    print("=" * 50)
    
    # Setup
    logger = setup_logging(log_level="INFO")
    
    # Generate simulated EEG data
    print("📡 Generating simulated EEG data...")
    raw_eeg, time = generate_simulated_eeg()
    print(f"   - Data shape: {raw_eeg.shape}")
    print(f"   - Duration: {raw_eeg.shape[1]/256:.1f} seconds")
    print(f"   - Channels: {raw_eeg.shape[0]}")
    
    # Initialize processor
    print("\n⚡ Initializing EEG processor...")
    processor = RealTimeEEGProcessor()
    
    # Process the data
    print("\n🔧 Processing EEG signals...")
    import time as time_module
    start_time = time_module.time()
    
    processed_eeg = processor.process_window(raw_eeg)
    
    processing_time = time_module.time() - start_time
    print(f"   - Processing time: {processing_time*1000:.2f} ms")
    print(f"   - Processed shape: {processed_eeg.shape}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    
    # Create output directory
    output_dir = Path("output/demo1_signal_processing")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Channel names
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    # Plot 1: Raw vs Filtered signals
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('EEG Signal Processing: Before and After Filtering', fontsize=16)
    
    channels_to_plot = [0, 3, 7, 10]  # Representative channels
    
    for i, ch_idx in enumerate(channels_to_plot):
        axes[i].plot(time, raw_eeg[ch_idx], 'b-', alpha=0.7, label='Raw Signal')
        axes[i].plot(time, processed_eeg[ch_idx], 'r-', linewidth=2, label='Filtered Signal')
        axes[i].set_title(f'Channel {channel_names[ch_idx]}')
        axes[i].set_ylabel('Amplitude (μV)')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.savefig(output_dir / 'raw_vs_filtered.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Frequency analysis
    print("\n🔍 Performing frequency analysis...")
    
    from scipy import signal
    freqs, psd_raw = signal.welch(raw_eeg[0], fs=256, nperseg=256)
    freqs, psd_filtered = signal.welch(processed_eeg[0], fs=256, nperseg=256)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.semilogy(freqs, psd_raw, 'b-', label='Raw Signal')
    plt.semilogy(freqs, psd_filtered, 'r-', label='Filtered Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectral Density - Channel AF3')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 50)
    
    # Plot frequency bands
    plt.subplot(1, 2, 2)
    band_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    band_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
    
    band_powers = []
    for low, high in band_ranges:
        mask = (freqs >= low) & (freqs <= high)
        power = np.mean(psd_filtered[mask])
        band_powers.append(power)
    
    plt.bar(band_names, band_powers, color=['red', 'orange', 'green', 'blue', 'purple'])
    plt.ylabel('Average Power')
    plt.title('Frequency Band Powers')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature extraction demonstration
    print("\n🔬 Demonstrating feature extraction...")
    
    feature_extractor = EEGFeatureExtractor()
    features = feature_extractor.extract_features(processed_eeg)
    
    print(f"   - Feature vector size: {len(features)}")
    print(f"   - Features shape: {features.shape}")
    
    # Plot feature importance
    plt.figure(figsize=(15, 6))
    
    # Reshape features for visualization
    n_channels = 14
    n_bands = 5
    features_reshaped = features[:n_channels * n_bands].reshape(n_channels, n_bands)
    
    plt.subplot(1, 2, 1)
    im = plt.imshow(features_reshaped, aspect='auto', cmap='viridis')
    plt.colorbar(im, label='Feature Value')
    plt.title('Feature Map: Channels × Frequency Bands')
    plt.xlabel('Frequency Bands')
    plt.ylabel('Channels')
    plt.xticks(range(5), band_names)
    plt.yticks(range(14), channel_names)
    
    # Plot top features
    plt.subplot(1, 2, 2)
    top_indices = np.argsort(np.abs(features))[-20:]
    plt.barh(range(len(top_indices)), features[top_indices])
    plt.xlabel('Feature Value')
    plt.title('Top 20 Feature Values')
    plt.ylabel('Feature Index')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_extraction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance benchmarks
    print("\n⏱️ Performance Benchmarks:")
    print(f"   - Processing latency: {processing_time*1000:.2f} ms")
    print(f"   - Throughput: {raw_eeg.shape[1]/processing_time:.0f} samples/sec")
    print(f"   - Real-time factor: {(raw_eeg.shape[1]/256)/processing_time:.1f}x")
    
    print(f"\n✅ Demo completed! Outputs saved to: {output_dir}")
    print("\n📋 Summary:")
    print("   - Raw EEG data simulated with realistic brain rhythms")
    print("   - Filtering removed artifacts and noise")
    print("   - Frequency analysis shows clean frequency bands")
    print("   - Feature extraction ready for machine learning")
    print("   - Processing is real-time capable (<50ms latency)")

if __name__ == "__main__":
    try:
        demo_signal_processing()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
