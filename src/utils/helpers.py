#!/usr/bin/env python3
"""
Utility Functions
=================

Common utility functions for the EEG-Robot control system.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
import time
from datetime import datetime

def setup_logging(log_dir: str = "logs", log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"system_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ValueError(f"Failed to load config {config_path}: {e}")

def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save config {config_path}: {e}")

def create_directory(path: str):
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)

def validate_eeg_data(data: np.ndarray, expected_channels: int = None) -> bool:
    """Validate EEG data format"""
    if not isinstance(data, np.ndarray):
        return False
    
    if data.ndim != 2:
        return False
    
    if expected_channels and data.shape[0] != expected_channels:
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        return False
    
    return True

def normalize_eeg_data(data: np.ndarray, method: str = 'z-score') -> np.ndarray:
    """Normalize EEG data"""
    if method == 'z-score':
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return (data - mean) / (std + 1e-8)
    
    elif method == 'min-max':
        min_val = np.min(data, axis=1, keepdims=True)
        max_val = np.max(data, axis=1, keepdims=True)
        return (data - min_val) / (max_val - min_val + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def calculate_signal_quality(data: np.ndarray, sampling_rate: int) -> Dict[str, float]:
    """Calculate signal quality metrics"""
    quality = {}
    
    # Signal-to-noise ratio (simplified)
    signal_power = np.mean(data**2, axis=1)
    noise_estimate = np.var(np.diff(data, axis=1), axis=1) / 2
    snr = 10 * np.log10(signal_power / (noise_estimate + 1e-8))
    quality['snr_db'] = float(np.mean(snr))
    
    # Standard deviation (should be in reasonable range for EEG)
    std_values = np.std(data, axis=1)
    quality['mean_std'] = float(np.mean(std_values))
    quality['std_range'] = (float(np.min(std_values)), float(np.max(std_values)))
    
    # Check for artifacts (very high amplitude)
    artifact_threshold = 100  # microvolts
    artifact_ratio = np.mean(np.abs(data) > artifact_threshold)
    quality['artifact_ratio'] = float(artifact_ratio)
    
    # Frequency content check
    from scipy.fft import fft, fftfreq
    fft_data = fft(data, axis=1)
    freqs = fftfreq(data.shape[1], 1/sampling_rate)
    
    # Power in different bands
    alpha_power = np.mean(np.abs(fft_data[:, (freqs >= 8) & (freqs <= 12)]))
    total_power = np.mean(np.abs(fft_data[:, freqs > 0]))
    quality['alpha_ratio'] = float(alpha_power / (total_power + 1e-8))
    
    return quality

def format_time_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def create_performance_report(stats: Dict[str, Any]) -> str:
    """Create a formatted performance report"""
    report = []
    report.append("=" * 50)
    report.append("PERFORMANCE REPORT")
    report.append("=" * 50)
    
    if 'system' in stats:
        system_stats = stats['system']
        report.append(f"System Uptime: {format_time_duration(system_stats.get('uptime', 0))}")
        report.append(f"Windows Processed: {system_stats.get('windows_processed', 0)}")
        report.append(f"Predictions Made: {system_stats.get('predictions_made', 0)}")
        report.append(f"Commands Sent: {system_stats.get('commands_sent', 0)}")
        report.append("")
    
    if 'eeg_processor' in stats:
        proc_stats = stats['eeg_processor']
        report.append(f"EEG Processing:")
        report.append(f"  Channels: {proc_stats.get('channels', 0)}")
        report.append(f"  Sampling Rate: {proc_stats.get('sampling_rate', 0)} Hz")
        report.append(f"  Buffer Size: {proc_stats.get('buffer_size', 0)}")
        report.append(f"  Processing Rate: {proc_stats.get('processing_rate', 0):.2f} windows/s")
        report.append("")
    
    if 'model_inference' in stats:
        model_stats = stats['model_inference']
        report.append(f"Model Inference:")
        report.append(f"  Model Loaded: {model_stats.get('model_loaded', False)}")
        report.append(f"  Inference Count: {model_stats.get('inference_count', 0)}")
        report.append(f"  Avg Inference Time: {model_stats.get('average_inference_time', 0):.4f}s")
        report.append(f"  Confidence Threshold: {model_stats.get('confidence_threshold', 0):.2f}")
        report.append("")
    
    if 'robot_controller' in stats:
        robot_stats = stats['robot_controller']
        report.append(f"Robot Control:")
        report.append(f"  Robot Active: {robot_stats.get('is_running', False)}")
        report.append(f"  Position: {robot_stats.get('position', (0, 0, 0))}")
        report.append(f"  Last Command: {robot_stats.get('last_command', 'none')}")
        report.append(f"  Safety Status: {robot_stats.get('safety_status', 'unknown')}")
        report.append("")
    
    if 'performance' in stats:
        perf_stats = stats['performance']
        report.append(f"Performance Metrics:")
        report.append(f"  Avg Processing Time: {perf_stats.get('avg_processing_time', 0):.4f}s")
        report.append(f"  Max Processing Time: {perf_stats.get('max_processing_time', 0):.4f}s")
        report.append(f"  Min Processing Time: {perf_stats.get('min_processing_time', 0):.4f}s")
    
    report.append("=" * 50)
    
    return "\n".join(report)

def export_data_to_csv(data: List[Dict[str, Any]], filename: str):
    """Export data to CSV file"""
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return True
    except Exception as e:
        logging.error(f"Failed to export data to CSV: {e}")
        return False

def load_edf_file(filepath: str) -> Dict[str, Any]:
    """Load EDF file and return data dictionary"""
    try:
        import mne
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        return {
            'data': raw.get_data(),
            'channel_names': raw.ch_names,
            'sampling_rate': raw.info['sfreq'],
            'duration': raw.times[-1],
            'n_channels': len(raw.ch_names),
            'n_samples': len(raw.times)
        }
    except Exception as e:
        logging.error(f"Failed to load EDF file {filepath}: {e}")
        return None

def bandpass_filter(data: np.ndarray, low_freq: float, high_freq: float, 
                   sampling_rate: int, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to data"""
    try:
        from scipy import signal
        
        nyquist = sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        
        # Apply filter to each channel
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = signal.filtfilt(b, a, data[ch])
        
        return filtered_data
    
    except Exception as e:
        logging.error(f"Bandpass filtering failed: {e}")
        return data

def calculate_frequency_bands(data: np.ndarray, sampling_rate: int, 
                            bands: Dict[str, tuple]) -> Dict[str, np.ndarray]:
    """Calculate power in frequency bands"""
    try:
        from scipy.fft import fft, fftfreq
        
        # Compute FFT
        fft_data = fft(data, axis=1)
        freqs = fftfreq(data.shape[1], 1/sampling_rate)
        
        # Only use positive frequencies
        positive_mask = freqs >= 0
        freqs_pos = freqs[positive_mask]
        fft_magnitude = np.abs(fft_data[:, positive_mask])
        
        # Calculate band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in bands.items():
            band_mask = (freqs_pos >= low_freq) & (freqs_pos <= high_freq)
            if np.any(band_mask):
                band_powers[band_name] = np.sum(fft_magnitude[:, band_mask], axis=1)
            else:
                band_powers[band_name] = np.zeros(data.shape[0])
        
        return band_powers
    
    except Exception as e:
        logging.error(f"Frequency band calculation failed: {e}")
        return {}

def check_system_requirements() -> Dict[str, bool]:
    """Check if system requirements are met"""
    requirements = {
        'python_version': sys.version_info >= (3, 7),
        'numpy': True,
        'scipy': True,
        'yaml': True,
        'pandas': True
    }
    
    # Check optional dependencies
    try:
        import torch
        requirements['pytorch'] = True
    except ImportError:
        requirements['pytorch'] = False
    
    try:
        import mne
        requirements['mne'] = True
    except ImportError:
        requirements['mne'] = False
    
    try:
        from cortex import Cortex
        requirements['emotiv_cortex'] = True
    except ImportError:
        requirements['emotiv_cortex'] = False
    
    return requirements

def print_system_info():
    """Print system information and requirements"""
    import sys
    import platform
    
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print(f"Processor: {platform.processor()}")
    print("")
    
    print("REQUIREMENTS CHECK:")
    requirements = check_system_requirements()
    for req, status in requirements.items():
        status_str = "✓ OK" if status else "✗ MISSING"
        print(f"  {req}: {status_str}")
    
    print("=" * 60)

if __name__ == "__main__":
    # Test utility functions
    print_system_info()
    
    # Test configuration loading
    try:
        config = load_config("config/pipeline.yaml")
        print(f"Successfully loaded pipeline config with {len(config)} sections")
    except Exception as e:
        print(f"Failed to load config: {e}")
    
    # Test EEG data validation
    test_data = np.random.randn(14, 1024)  # 14 channels, 1024 samples
    print(f"Test EEG data validation: {validate_eeg_data(test_data, 14)}")
    
    # Test signal quality calculation
    quality = calculate_signal_quality(test_data, 256)
    print(f"Signal quality metrics: {quality}")
    
    print("Utility functions test completed!")
