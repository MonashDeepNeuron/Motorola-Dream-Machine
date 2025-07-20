#!/usr/bin/env python3
"""
Real-time EEG Signal Processor
==============================

This module provides real-time EEG signal processing capabilities including
filtering, feature extraction, and windowing for the Motorola Dream Machine.
"""

import numpy as np
import mne
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional, Any
import yaml
import logging
from collections import deque
import time

class RealTimeEEGProcessor:
    """Real-time EEG signal processor for live streaming data"""
    
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        """Initialize processor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get EEG processing config
        self.eeg_config = self.config['eeg_processing']
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Sampling parameters
        self.sampling_rate = self.eeg_config['sampling_rate']
        self.window_length = self.eeg_config['window_length']
        self.overlap = self.eeg_config['overlap']
        
        # Calculate window parameters
        self.window_samples = int(self.window_length * self.sampling_rate)
        self.overlap_samples = int(self.overlap * self.window_samples)
        self.step_samples = self.window_samples - self.overlap_samples
        
        # Data buffers
        self.raw_buffer = deque(maxlen=self.window_samples * 2)  # Double buffer
        self.channel_names = []
        self.n_channels = 0
        
        # Filter parameters
        self.filter_config = self.eeg_config['filters']
        self.lowpass_freq = self.filter_config['lowpass']
        self.highpass_freq = self.filter_config['highpass']
        self.notch_freq = self.filter_config['notch']
        
        # Frequency bands for feature extraction
        self.bands = self.eeg_config['frequency_bands']
        
        # Initialize filters
        self._initialize_filters()
        
        # Statistics tracking
        self.processed_windows = 0
        self.last_process_time = time.time()
        
        self.logger.info(f"Initialized EEG processor: {self.window_samples} samples/window, {self.sampling_rate} Hz")
    
    def _initialize_filters(self):
        """Initialize digital filters for real-time processing"""
        try:
            # Bandpass filter (highpass + lowpass)
            self.bandpass_sos = signal.butter(
                4, [self.highpass_freq, self.lowpass_freq], 
                btype='band', fs=self.sampling_rate, output='sos'
            )
            
            # Notch filter for line noise
            if self.notch_freq:
                self.notch_sos = signal.iirnotch(
                    self.notch_freq, Q=30, fs=self.sampling_rate
                )
            else:
                self.notch_sos = None
            
            # Initialize filter states for continuous processing
            self.bandpass_zi = None
            self.notch_zi = None
            
            self.logger.info(f"Filters initialized: {self.highpass_freq}-{self.lowpass_freq} Hz bandpass, {self.notch_freq} Hz notch")
            
        except Exception as e:
            self.logger.error(f"Filter initialization failed: {e}")
            raise
    
    def set_channels(self, channel_names: List[str]):
        """Set channel names and initialize channel-specific processing"""
        self.channel_names = channel_names
        self.n_channels = len(channel_names)
        
        # Initialize filter states for each channel
        if self.n_channels > 0:
            self.bandpass_zi = np.zeros((self.n_channels, self.bandpass_sos.shape[0], 2))
            if self.notch_sos is not None:
                self.notch_zi = np.zeros((self.n_channels, 2))
        
        self.logger.info(f"Set {self.n_channels} channels: {channel_names}")
    
    def add_sample(self, eeg_data: np.ndarray, timestamp: float = None):
        """Add new EEG sample to processing buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        # Validate input
        if len(eeg_data) != self.n_channels:
            self.logger.warning(f"Channel mismatch: expected {self.n_channels}, got {len(eeg_data)}")
            return
        
        # Add to buffer with timestamp
        self.raw_buffer.append({
            'data': eeg_data.copy(),
            'timestamp': timestamp
        })
    
    def can_process_window(self) -> bool:
        """Check if enough data is available for processing"""
        return len(self.raw_buffer) >= self.window_samples
    
    def process_window(self) -> Optional[Dict[str, Any]]:
        """Process a complete window of EEG data"""
        if not self.can_process_window():
            return None
        
        try:
            # Extract window data
            window_data = self._extract_window()
            
            if window_data is None:
                return None
            
            # Apply preprocessing
            filtered_data = self._apply_filters(window_data['data'])
            
            # Extract features
            features = self._extract_features(filtered_data)
            
            # Create result
            result = {
                'timestamp': window_data['timestamp'],
                'raw_data': window_data['data'],
                'filtered_data': filtered_data,
                'features': features,
                'channel_names': self.channel_names,
                'window_id': self.processed_windows
            }
            
            self.processed_windows += 1
            self.last_process_time = time.time()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Window processing failed: {e}")
            return None
    
    def _extract_window(self) -> Optional[Dict[str, Any]]:
        """Extract a window of data from buffer"""
        if len(self.raw_buffer) < self.window_samples:
            return None
        
        # Get latest window_samples
        window_samples = list(self.raw_buffer)[-self.window_samples:]
        
        # Extract data and timestamps
        data_matrix = np.array([sample['data'] for sample in window_samples])
        timestamps = [sample['timestamp'] for sample in window_samples]
        
        # Transpose to (channels, samples)
        data_matrix = data_matrix.T
        
        return {
            'data': data_matrix,
            'timestamps': timestamps,
            'timestamp': timestamps[-1],  # Use last timestamp
            'duration': timestamps[-1] - timestamps[0]
        }
    
    def _apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply digital filters to EEG data"""
        filtered_data = data.copy()
        
        try:
            # Apply bandpass filter to each channel
            for ch in range(self.n_channels):
                if self.bandpass_zi is not None:
                    filtered_data[ch], self.bandpass_zi[ch] = signal.sosfilt(
                        self.bandpass_sos, filtered_data[ch], zi=self.bandpass_zi[ch]
                    )
                else:
                    filtered_data[ch] = signal.sosfilt(self.bandpass_sos, filtered_data[ch])
            
            # Apply notch filter if configured
            if self.notch_sos is not None:
                for ch in range(self.n_channels):
                    if self.notch_zi is not None:
                        filtered_data[ch], self.notch_zi[ch] = signal.lfilter(
                            self.notch_sos[0], self.notch_sos[1], 
                            filtered_data[ch], zi=self.notch_zi[ch]
                        )
                    else:
                        filtered_data[ch] = signal.lfilter(
                            self.notch_sos[0], self.notch_sos[1], filtered_data[ch]
                        )
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}")
            return data  # Return unfiltered data as fallback
    
    def _extract_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract features from filtered EEG data"""
        features = {}
        
        try:
            # Time domain features
            features['time_domain'] = self._extract_time_features(data)
            
            # Frequency domain features
            features['frequency_domain'] = self._extract_frequency_features(data)
            
            # Statistical features
            features['statistical'] = self._extract_statistical_features(data)
            
            # Power spectral density
            features['psd'] = self._extract_psd_features(data)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _extract_time_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract time domain features"""
        return {
            'mean': np.mean(data, axis=1),
            'std': np.std(data, axis=1),
            'var': np.var(data, axis=1),
            'rms': np.sqrt(np.mean(data**2, axis=1)),
            'peak_to_peak': np.ptp(data, axis=1),
            'skewness': self._skewness(data, axis=1),
            'kurtosis': self._kurtosis(data, axis=1)
        }
    
    def _extract_frequency_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract frequency domain features"""
        # Compute FFT for each channel
        fft_data = fft(data, axis=1)
        freqs = fftfreq(data.shape[1], 1/self.sampling_rate)
        
        # Only use positive frequencies
        positive_freqs = freqs[:len(freqs)//2]
        fft_magnitude = np.abs(fft_data[:, :len(freqs)//2])
        
        # Extract band powers
        band_powers = {}
        for band_name, (low_freq, high_freq) in self.bands.items():
            band_mask = (positive_freqs >= low_freq) & (positive_freqs <= high_freq)
            band_powers[f'{band_name}_power'] = np.sum(fft_magnitude[:, band_mask], axis=1)
            band_powers[f'{band_name}_relative'] = (
                band_powers[f'{band_name}_power'] / np.sum(fft_magnitude, axis=1)
            )
        
        return {
            'band_powers': band_powers,
            'dominant_frequency': positive_freqs[np.argmax(fft_magnitude, axis=1)],
            'spectral_centroid': np.sum(positive_freqs * fft_magnitude, axis=1) / np.sum(fft_magnitude, axis=1),
            'spectral_bandwidth': self._spectral_bandwidth(positive_freqs, fft_magnitude)
        }
    
    def _extract_statistical_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract statistical features"""
        return {
            'hjorth_activity': np.var(data, axis=1),
            'hjorth_mobility': self._hjorth_mobility(data),
            'hjorth_complexity': self._hjorth_complexity(data),
            'zero_crossings': self._zero_crossings(data),
            'line_length': np.sum(np.abs(np.diff(data, axis=1)), axis=1)
        }
    
    def _extract_psd_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract power spectral density features"""
        psd_features = {}
        
        for ch in range(self.n_channels):
            freqs, psd = signal.welch(
                data[ch], fs=self.sampling_rate, 
                nperseg=min(256, len(data[ch])//2)
            )
            
            # Band-specific PSD
            for band_name, (low_freq, high_freq) in self.bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                psd_features[f'ch{ch}_{band_name}_psd'] = np.mean(psd[band_mask])
        
        return psd_features
    
    def _skewness(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Calculate skewness"""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return np.mean(((data - mean) / std)**3, axis=axis)
    
    def _kurtosis(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Calculate kurtosis"""
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        return np.mean(((data - mean) / std)**4, axis=axis) - 3
    
    def _hjorth_mobility(self, data: np.ndarray) -> np.ndarray:
        """Calculate Hjorth mobility parameter"""
        diff1 = np.diff(data, axis=1)
        return np.sqrt(np.var(diff1, axis=1) / np.var(data, axis=1))
    
    def _hjorth_complexity(self, data: np.ndarray) -> np.ndarray:
        """Calculate Hjorth complexity parameter"""
        diff1 = np.diff(data, axis=1)
        diff2 = np.diff(diff1, axis=1)
        
        mobility1 = np.sqrt(np.var(diff1, axis=1) / np.var(data, axis=1))
        mobility2 = np.sqrt(np.var(diff2, axis=1) / np.var(diff1, axis=1))
        
        return mobility2 / mobility1
    
    def _zero_crossings(self, data: np.ndarray) -> np.ndarray:
        """Count zero crossings"""
        zero_crossings = np.zeros(data.shape[0])
        for ch in range(data.shape[0]):
            zero_crossings[ch] = np.sum(np.diff(np.sign(data[ch])) != 0)
        return zero_crossings
    
    def _spectral_bandwidth(self, freqs: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
        """Calculate spectral bandwidth"""
        centroid = np.sum(freqs * magnitude, axis=1) / np.sum(magnitude, axis=1)
        bandwidth = np.zeros(magnitude.shape[0])
        
        for ch in range(magnitude.shape[0]):
            bandwidth[ch] = np.sqrt(
                np.sum(((freqs - centroid[ch])**2) * magnitude[ch]) / np.sum(magnitude[ch])
            )
        
        return bandwidth
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status information"""
        return {
            'channels': self.n_channels,
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'window_samples': self.window_samples,
            'buffer_size': len(self.raw_buffer),
            'processed_windows': self.processed_windows,
            'can_process': self.can_process_window(),
            'last_process_time': self.last_process_time,
            'processing_rate': self.processed_windows / max(1, time.time() - self.last_process_time) if self.processed_windows > 0 else 0
        }
    
    def reset_buffers(self):
        """Reset processing buffers"""
        self.raw_buffer.clear()
        self.processed_windows = 0
        
        # Reset filter states
        if self.bandpass_zi is not None:
            self.bandpass_zi.fill(0)
        if self.notch_zi is not None:
            self.notch_zi.fill(0)
        
        self.logger.info("Reset processing buffers")

def main():
    """Test the EEG processor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Real-time EEG Processor")
    parser.add_argument('--config', default='config/pipeline.yaml', help='Config file path')
    parser.add_argument('--duration', type=int, default=10, help='Test duration (seconds)')
    parser.add_argument('--channels', nargs='+', 
                       default=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
                       help='EEG channel names')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RealTimeEEGProcessor(args.config)
    processor.set_channels(args.channels)
    
    print(f"Testing EEG processor for {args.duration} seconds...")
    print(f"Status: {processor.get_status()}")
    
    # Simulate real-time data
    sampling_rate = processor.sampling_rate
    n_channels = len(args.channels)
    
    start_time = time.time()
    sample_count = 0
    processed_count = 0
    
    try:
        while time.time() - start_time < args.duration:
            # Generate synthetic EEG sample
            t = time.time()
            eeg_sample = np.random.randn(n_channels) * 20  # μV scale
            
            # Add some realistic signal components
            for ch in range(n_channels):
                alpha = 10 * np.sin(2 * np.pi * 10 * t + ch * 0.1)  # 10 Hz alpha
                beta = 5 * np.sin(2 * np.pi * 20 * t + ch * 0.2)    # 20 Hz beta
                eeg_sample[ch] += alpha + beta
            
            # Add sample to processor
            processor.add_sample(eeg_sample, t)
            sample_count += 1
            
            # Process if possible
            if processor.can_process_window():
                result = processor.process_window()
                if result:
                    processed_count += 1
                    features = result['features']
                    
                    print(f"Window {processed_count}: "
                          f"Alpha power: {np.mean(features['frequency_domain']['band_powers']['alpha_power']):.2f}, "
                          f"Beta power: {np.mean(features['frequency_domain']['band_powers']['beta_power']):.2f}")
            
            # Simulate sampling rate
            time.sleep(1.0 / sampling_rate)
        
        # Final status
        print(f"\nProcessing completed!")
        print(f"Samples collected: {sample_count}")
        print(f"Windows processed: {processed_count}")
        print(f"Final status: {processor.get_status()}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    print("Done!")

if __name__ == "__main__":
    main()
