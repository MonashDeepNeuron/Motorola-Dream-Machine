#!/usr/bin/env python3
"""
Demo 2: Feature Extraction Deep Dive
===================================

Understand what the ML model actually "sees" from EEG signals.

Usage:
    python3 demos/demo_feature_extraction.py

Output:
    - 5 frequency band power maps
    - Channel connectivity matrices  
    - Feature importance rankings
    - Real-time feature streaming
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.eeg.features import EEGFeatureExtractor
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

def generate_realistic_eeg(duration=4, sampling_rate=256, channels=14):
    """Generate realistic EEG data with brain-like patterns."""
    samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, samples)
    
    eeg_data = np.zeros((channels, samples))
    
    # Channel names (standard 10-20 system)
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    for ch in range(channels):
        # Different brain regions have different dominant frequencies
        if 'F' in channel_names[ch] or 'AF' in channel_names[ch]:  # Frontal
            # More beta (concentration) and gamma (cognitive processing)
            alpha = 0.3 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
            beta = 0.7 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
            gamma = 0.4 * np.sin(2 * np.pi * 35 * time + np.random.rand() * 2 * np.pi)
            theta = 0.2 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
            
        elif 'O' in channel_names[ch]:  # Occipital (visual)
            # Strong alpha rhythm when eyes closed
            alpha = 0.8 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
            beta = 0.2 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
            gamma = 0.1 * np.sin(2 * np.pi * 35 * time + np.random.rand() * 2 * np.pi)
            theta = 0.3 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
            
        elif 'T' in channel_names[ch] or 'C' in channel_names[ch]:  # Motor cortex
            # More beta for motor planning
            alpha = 0.4 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
            beta = 0.6 * np.sin(2 * np.pi * 22 * time + np.random.rand() * 2 * np.pi)
            gamma = 0.2 * np.sin(2 * np.pi * 35 * time + np.random.rand() * 2 * np.pi)
            theta = 0.3 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
            
        else:  # Parietal
            # Balanced activity
            alpha = 0.5 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
            beta = 0.4 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
            gamma = 0.2 * np.sin(2 * np.pi * 35 * time + np.random.rand() * 2 * np.pi)
            theta = 0.4 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
        
        # Add delta rhythm (sleep/unconscious)
        delta = 0.6 * np.sin(2 * np.pi * 2 * time + np.random.rand() * 2 * np.pi)
        
        # Random noise
        noise = 0.1 * np.random.randn(samples)
        
        # Combine all components
        eeg_data[ch] = delta + theta + alpha + beta + gamma + noise
    
    return eeg_data, channel_names

def demo_feature_extraction():
    """Demonstrate EEG feature extraction in detail."""
    print("🔬 Demo 2: Feature Extraction Deep Dive")
    print("=" * 50)
    
    # Setup
    logger = setup_logging(log_level="INFO")
    
    # Generate EEG data
    print("📡 Generating realistic EEG data...")
    eeg_data, channel_names = generate_realistic_eeg()
    print(f"   - Data shape: {eeg_data.shape}")
    print(f"   - Channels: {channel_names}")
    
    # Initialize feature extractor
    print("\n🔬 Initializing feature extractor...")
    feature_extractor = EEGFeatureExtractor()
    
    # Extract features
    print("\n🧮 Extracting features...")
    features = feature_extractor.extract_features(eeg_data)
    
    # Convert features to numpy array if it's not already
    if isinstance(features, dict):
        # Handle case where features is a dictionary
        features_array = []
        for key, value in features.items():
            if isinstance(value, (list, np.ndarray)):
                features_array.extend(np.array(value).flatten())
            else:
                features_array.append(value)
        features = np.array(features_array)
    elif not isinstance(features, np.ndarray):
        features = np.array(features)
    
    print(f"   - Feature vector size: {len(features)}")
    print(f"   - Features shape: {features.shape}")
    
    # Create output directory
    output_dir = Path("output/demo2_feature_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze frequency band powers
    print("\n📊 Analyzing frequency band powers...")
    
    n_channels = len(channel_names)
    n_bands = 5
    band_names = ['Delta (0.5-4 Hz)', 'Theta (4-8 Hz)', 'Alpha (8-13 Hz)', 'Beta (13-30 Hz)', 'Gamma (30-40 Hz)']
    
    # Reshape features (assuming first n_channels*n_bands features are band powers)
    band_powers = features[:n_channels * n_bands].reshape(n_channels, n_bands)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('EEG Feature Extraction Analysis', fontsize=16)
    
    # 1. Band power heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(band_powers, aspect='auto', cmap='viridis')
    ax1.set_title('Frequency Band Powers by Channel')
    ax1.set_xlabel('Frequency Bands')
    ax1.set_ylabel('EEG Channels')
    ax1.set_xticks(range(n_bands))
    ax1.set_xticklabels([name.split()[0] for name in band_names], rotation=45)
    ax1.set_yticks(range(n_channels))
    ax1.set_yticklabels(channel_names)
    plt.colorbar(im1, ax=ax1, label='Power')
    
    # 2. Average power per frequency band
    ax2 = axes[0, 1]
    avg_powers = np.mean(band_powers, axis=0)
    bars = ax2.bar(range(n_bands), avg_powers, color=['red', 'orange', 'green', 'blue', 'purple'])
    ax2.set_title('Average Power per Frequency Band')
    ax2.set_xlabel('Frequency Bands')
    ax2.set_ylabel('Average Power')
    ax2.set_xticks(range(n_bands))
    ax2.set_xticklabels([name.split()[0] for name in band_names], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_powers[i]:.3f}', ha='center', va='bottom')
    
    # 3. Channel power distribution
    ax3 = axes[0, 2]
    channel_powers = np.sum(band_powers, axis=1)
    ax3.barh(range(n_channels), channel_powers)
    ax3.set_title('Total Power per Channel')
    ax3.set_xlabel('Total Power')
    ax3.set_ylabel('EEG Channels')
    ax3.set_yticks(range(n_channels))
    ax3.set_yticklabels(channel_names)
    
    # 4. Feature importance (top features)
    ax4 = axes[1, 0]
    top_indices = np.argsort(np.abs(features))[-20:]
    top_features = features[top_indices]
    ax4.barh(range(len(top_indices)), top_features)
    ax4.set_title('Top 20 Feature Values')
    ax4.set_xlabel('Feature Value')
    ax4.set_ylabel('Feature Index')
    
    # 5. Brain region analysis
    ax5 = axes[1, 1]
    # Group channels by brain region
    frontal_idx = [i for i, name in enumerate(channel_names) if 'F' in name or 'AF' in name]
    motor_idx = [i for i, name in enumerate(channel_names) if 'C' in name or 'T' in name]
    parietal_idx = [i for i, name in enumerate(channel_names) if 'P' in name]
    occipital_idx = [i for i, name in enumerate(channel_names) if 'O' in name]
    
    region_powers = []
    region_names = ['Frontal', 'Motor', 'Parietal', 'Occipital']
    
    for indices in [frontal_idx, motor_idx, parietal_idx, occipital_idx]:
        if indices:
            region_power = np.mean(channel_powers[indices])
            region_powers.append(region_power)
        else:
            region_powers.append(0)
    
    wedges, texts, autotexts = ax5.pie(region_powers, labels=region_names, autopct='%1.1f%%')
    ax5.set_title('Power Distribution by Brain Region')
    
    # 6. Feature correlations
    ax6 = axes[1, 2]
    # Create a subset of features for correlation analysis
    feature_subset = features[:min(50, len(features))].reshape(-1, 1)
    correlation_matrix = np.corrcoef(feature_subset.T)
    
    if correlation_matrix.size > 1:
        im6 = ax6.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax6.set_title('Feature Correlations (Sample)')
        plt.colorbar(im6, ax=ax6, label='Correlation')
    else:
        ax6.text(0.5, 0.5, 'Insufficient features\nfor correlation analysis', 
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Feature Correlations')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature interpretation
    print("\n🧠 Feature Interpretation:")
    print("   - Feature vector contains information about:")
    print("     • Power in each frequency band (Delta, Theta, Alpha, Beta, Gamma)")
    print("     • Spatial patterns across brain regions")
    print("     • Cross-channel relationships")
    print("     • Temporal dynamics")
    
    # Most informative features
    print(f"\n📈 Most Informative Features:")
    top_feature_indices = np.argsort(np.abs(features))[-10:]
    for i, idx in enumerate(reversed(top_feature_indices)):
        feature_type = ""
        if idx < n_channels * n_bands:
            ch_idx = idx // n_bands
            band_idx = idx % n_bands
            feature_type = f"{channel_names[ch_idx]} {band_names[band_idx].split()[0]}"
        else:
            feature_type = f"Advanced feature {idx}"
        
        print(f"     {i+1:2d}. Feature {idx:3d}: {features[idx]:7.4f} ({feature_type})")
    
    # Save feature data
    feature_data = {
        'features': features.tolist(),
        'band_powers': band_powers.tolist(),
        'channel_names': channel_names,
        'band_names': band_names,
        'avg_powers': avg_powers.tolist(),
        'channel_powers': channel_powers.tolist()
    }
    
    import json
    with open(output_dir / 'feature_data.json', 'w') as f:
        json.dump(feature_data, f, indent=2)
    
    print(f"\n✅ Demo completed!")
    print(f"📁 Outputs saved to: {output_dir}")
    print("\n📋 Key Insights:")
    print("   - EEG features capture brain activity patterns")
    print("   - Different brain regions show different frequency signatures")
    print("   - Features are ready for machine learning classification")
    print("   - This is what the ML model 'sees' when making predictions")

if __name__ == "__main__":
    try:
        demo_feature_extraction()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
