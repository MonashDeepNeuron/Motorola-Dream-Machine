#!/usr/bin/env python3
"""
EEG Feature Extraction Module
=============================

This module provides comprehensive feature extraction capabilities for EEG signals,
optimized for real-time brain-computer interface applications.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    sampling_rate: int
    frequency_bands: Dict[str, Tuple[float, float]]
    time_features: List[str]
    frequency_features: List[str]
    connectivity_features: List[str]
    normalization: str = 'none'  # 'none', 'z-score', 'min-max'

class EEGFeatureExtractor:
    """Comprehensive EEG feature extraction for BCI applications"""
    
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        """Initialize feature extractor with configuration"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract EEG processing configuration
        eeg_config = config_data['eeg_processing']
        
        self.config = FeatureConfig(
            sampling_rate=eeg_config['sampling_rate'],
            frequency_bands=eeg_config['frequency_bands'],
            time_features=['mean', 'std', 'var', 'rms', 'skewness', 'kurtosis', 'hjorth'],
            frequency_features=['band_power', 'relative_power', 'peak_frequency', 'spectral_entropy'],
            connectivity_features=['coherence', 'phase_coupling'],
            normalization=eeg_config.get('normalization', 'none')
        )
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Feature cache for efficiency
        self._feature_cache = {}
        
        self.logger.info(f"Initialized EEG feature extractor: {len(self.config.frequency_bands)} frequency bands")
    
    def extract_features(self, data: np.ndarray, channel_names: List[str] = None) -> Dict[str, Any]:
        """
        Extract comprehensive features from EEG data
        
        Args:
            data: EEG data array (channels x samples)
            channel_names: List of channel names
        
        Returns:
            Dictionary containing extracted features
        """
        if data.ndim != 2:
            raise ValueError("Data must be 2D array (channels x samples)")
        
        n_channels, n_samples = data.shape
        
        if channel_names is None:
            channel_names = [f'Ch{i+1}' for i in range(n_channels)]
        
        features = {
            'metadata': {
                'n_channels': n_channels,
                'n_samples': n_samples,
                'duration': n_samples / self.config.sampling_rate,
                'channel_names': channel_names,
                'sampling_rate': self.config.sampling_rate
            }
        }
        
        try:
            # Time domain features
            if 'time_domain' not in features:
                features['time_domain'] = {}
            
            for feature_type in self.config.time_features:
                if feature_type == 'hjorth':
                    hjorth_features = self._extract_hjorth_parameters(data)
                    features['time_domain'].update(hjorth_features)
                else:
                    features['time_domain'][feature_type] = self._extract_time_feature(data, feature_type)
            
            # Frequency domain features
            features['frequency_domain'] = self._extract_frequency_features(data)
            
            # Connectivity features
            if len(channel_names) > 1:
                features['connectivity'] = self._extract_connectivity_features(data)
            
            # Advanced features
            features['advanced'] = self._extract_advanced_features(data)
            
            # Normalize features if requested
            if self.config.normalization != 'none':
                features = self._normalize_features(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return features
    
    def _extract_time_feature(self, data: np.ndarray, feature_type: str) -> np.ndarray:
        """Extract specific time domain feature"""
        if feature_type == 'mean':
            return np.mean(data, axis=1)
        elif feature_type == 'std':
            return np.std(data, axis=1)
        elif feature_type == 'var':
            return np.var(data, axis=1)
        elif feature_type == 'rms':
            return np.sqrt(np.mean(data**2, axis=1))
        elif feature_type == 'skewness':
            return self._calculate_skewness(data)
        elif feature_type == 'kurtosis':
            return self._calculate_kurtosis(data)
        else:
            self.logger.warning(f"Unknown time feature: {feature_type}")
            return np.zeros(data.shape[0])
    
    def _extract_hjorth_parameters(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Hjorth parameters (Activity, Mobility, Complexity)"""
        activity = np.var(data, axis=1)
        
        # First derivative
        diff1 = np.diff(data, axis=1)
        mobility = np.sqrt(np.var(diff1, axis=1) / activity)
        
        # Second derivative
        diff2 = np.diff(diff1, axis=1)
        mobility2 = np.sqrt(np.var(diff2, axis=1) / np.var(diff1, axis=1))
        complexity = mobility2 / mobility
        
        return {
            'hjorth_activity': activity,
            'hjorth_mobility': mobility,
            'hjorth_complexity': complexity
        }
    
    def _extract_frequency_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract comprehensive frequency domain features"""
        freq_features = {}
        
        # Compute power spectral density for each channel
        freqs_list = []
        psd_list = []
        
        for ch in range(data.shape[0]):
            freqs, psd = signal.welch(
                data[ch], 
                fs=self.config.sampling_rate,
                nperseg=min(256, data.shape[1]//4),
                noverlap=None
            )
            freqs_list.append(freqs)
            psd_list.append(psd)
        
        # Band power features
        band_powers = self._extract_band_powers(freqs_list, psd_list)
        freq_features.update(band_powers)
        
        # Spectral features
        spectral_features = self._extract_spectral_features(freqs_list, psd_list)
        freq_features.update(spectral_features)
        
        # FFT-based features
        fft_features = self._extract_fft_features(data)
        freq_features.update(fft_features)
        
        return freq_features
    
    def _extract_band_powers(self, freqs_list: List[np.ndarray], psd_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract power in specific frequency bands"""
        n_channels = len(psd_list)
        band_features = {}
        
        # Initialize band power arrays
        for band_name in self.config.frequency_bands.keys():
            band_features[f'{band_name}_power'] = np.zeros(n_channels)
            band_features[f'{band_name}_relative_power'] = np.zeros(n_channels)
        
        # Calculate band powers for each channel
        for ch in range(n_channels):
            freqs = freqs_list[ch]
            psd = psd_list[ch]
            total_power = np.sum(psd)
            
            for band_name, (low_freq, high_freq) in self.config.frequency_bands.items():
                # Find frequency indices
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                
                if np.any(band_mask):
                    # Absolute band power
                    band_power = np.sum(psd[band_mask])
                    band_features[f'{band_name}_power'][ch] = band_power
                    
                    # Relative band power
                    if total_power > 0:
                        band_features[f'{band_name}_relative_power'][ch] = band_power / total_power
        
        return band_features
    
    def _extract_spectral_features(self, freqs_list: List[np.ndarray], psd_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract spectral characteristics"""
        n_channels = len(psd_list)
        
        spectral_features = {
            'peak_frequency': np.zeros(n_channels),
            'spectral_centroid': np.zeros(n_channels),
            'spectral_bandwidth': np.zeros(n_channels),
            'spectral_entropy': np.zeros(n_channels),
            'spectral_rolloff': np.zeros(n_channels)
        }
        
        for ch in range(n_channels):
            freqs = freqs_list[ch]
            psd = psd_list[ch]
            
            if len(psd) > 0 and np.sum(psd) > 0:
                # Peak frequency
                spectral_features['peak_frequency'][ch] = freqs[np.argmax(psd)]
                
                # Spectral centroid
                spectral_features['spectral_centroid'][ch] = np.sum(freqs * psd) / np.sum(psd)
                
                # Spectral bandwidth
                centroid = spectral_features['spectral_centroid'][ch]
                spectral_features['spectral_bandwidth'][ch] = np.sqrt(
                    np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd)
                )
                
                # Spectral entropy
                psd_norm = psd / np.sum(psd)
                psd_norm = psd_norm[psd_norm > 0]  # Remove zeros
                spectral_features['spectral_entropy'][ch] = entropy(psd_norm)
                
                # Spectral rolloff (95% of energy)
                cumsum_psd = np.cumsum(psd)
                rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
                if len(rolloff_idx) > 0:
                    spectral_features['spectral_rolloff'][ch] = freqs[rolloff_idx[0]]
        
        return spectral_features
    
    def _extract_fft_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract FFT-based features"""
        fft_data = fft(data, axis=1)
        freqs = fftfreq(data.shape[1], 1/self.config.sampling_rate)
        
        # Use only positive frequencies
        positive_mask = freqs >= 0
        freqs_pos = freqs[positive_mask]
        fft_magnitude = np.abs(fft_data[:, positive_mask])
        
        fft_features = {
            'fft_mean': np.mean(fft_magnitude, axis=1),
            'fft_std': np.std(fft_magnitude, axis=1),
            'fft_max': np.max(fft_magnitude, axis=1),
            'fft_sum': np.sum(fft_magnitude, axis=1)
        }
        
        return fft_features
    
    def _extract_connectivity_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract connectivity features between channels"""
        n_channels = data.shape[0]
        connectivity_features = {}
        
        # Cross-correlation
        cross_corr = self._calculate_cross_correlation(data)
        connectivity_features['cross_correlation'] = cross_corr
        
        # Coherence
        coherence_matrix = self._calculate_coherence(data)
        connectivity_features['coherence_matrix'] = coherence_matrix
        connectivity_features['mean_coherence'] = np.mean(coherence_matrix[np.triu_indices(n_channels, k=1)])
        
        # Phase locking value
        plv_matrix = self._calculate_phase_locking_value(data)
        connectivity_features['plv_matrix'] = plv_matrix
        connectivity_features['mean_plv'] = np.mean(plv_matrix[np.triu_indices(n_channels, k=1)])
        
        return connectivity_features
    
    def _extract_advanced_features(self, data: np.ndarray) -> Dict[str, Any]:
        """Extract advanced features"""
        advanced_features = {}
        
        # Zero crossings
        advanced_features['zero_crossings'] = self._calculate_zero_crossings(data)
        
        # Line length
        advanced_features['line_length'] = self._calculate_line_length(data)
        
        # Sample entropy
        advanced_features['sample_entropy'] = self._calculate_sample_entropy(data)
        
        # Fractal dimension
        advanced_features['fractal_dimension'] = self._calculate_fractal_dimension(data)
        
        # Energy
        advanced_features['energy'] = np.sum(data**2, axis=1)
        
        # Peak-to-peak amplitude
        advanced_features['peak_to_peak'] = np.ptp(data, axis=1)
        
        return advanced_features
    
    def _calculate_skewness(self, data: np.ndarray) -> np.ndarray:
        """Calculate skewness for each channel"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return np.mean(((data - mean) / (std + 1e-8))**3, axis=1)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Calculate kurtosis for each channel"""
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        return np.mean(((data - mean) / (std + 1e-8))**4, axis=1) - 3
    
    def _calculate_cross_correlation(self, data: np.ndarray) -> np.ndarray:
        """Calculate cross-correlation matrix"""
        n_channels = data.shape[0]
        corr_matrix = np.corrcoef(data)
        return corr_matrix
    
    def _calculate_coherence(self, data: np.ndarray) -> np.ndarray:
        """Calculate coherence matrix between channels"""
        n_channels = data.shape[0]
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    coherence_matrix[i, j] = 1.0
                else:
                    freqs, coherence = signal.coherence(
                        data[i], data[j], 
                        fs=self.config.sampling_rate,
                        nperseg=min(256, data.shape[1]//4)
                    )
                    # Use mean coherence across frequencies
                    mean_coherence = np.mean(coherence)
                    coherence_matrix[i, j] = mean_coherence
                    coherence_matrix[j, i] = mean_coherence
        
        return coherence_matrix
    
    def _calculate_phase_locking_value(self, data: np.ndarray) -> np.ndarray:
        """Calculate phase locking value matrix"""
        n_channels = data.shape[0]
        plv_matrix = np.zeros((n_channels, n_channels))
        
        # Hilbert transform to get instantaneous phase
        analytic_signals = signal.hilbert(data, axis=1)
        phases = np.angle(analytic_signals)
        
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    plv_matrix[i, j] = 1.0
                else:
                    phase_diff = phases[i] - phases[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_matrix[i, j] = plv
                    plv_matrix[j, i] = plv
        
        return plv_matrix
    
    def _calculate_zero_crossings(self, data: np.ndarray) -> np.ndarray:
        """Calculate zero crossings for each channel"""
        zero_crossings = np.zeros(data.shape[0])
        for ch in range(data.shape[0]):
            zero_crossings[ch] = np.sum(np.diff(np.sign(data[ch])) != 0)
        return zero_crossings
    
    def _calculate_line_length(self, data: np.ndarray) -> np.ndarray:
        """Calculate line length for each channel"""
        return np.sum(np.abs(np.diff(data, axis=1)), axis=1)
    
    def _calculate_sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> np.ndarray:
        """Calculate sample entropy for each channel"""
        entropies = np.zeros(data.shape[0])
        
        for ch in range(data.shape[0]):
            signal_data = data[ch]
            N = len(signal_data)
            
            # Normalize r to signal standard deviation
            r_norm = r * np.std(signal_data)
            
            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([signal_data[i:i + m] for i in range(N - m + 1)])
                C = np.zeros(N - m + 1)
                
                for i in range(N - m + 1):
                    template_i = patterns[i]
                    for j in range(N - m + 1):
                        if _maxdist(template_i, patterns[j], N, m) <= r_norm:
                            C[i] += 1.0
                
                phi = np.mean([np.log(c / (N - m + 1.0)) for c in C if c > 0])
                return phi
            
            try:
                entropies[ch] = _phi(m) - _phi(m + 1)
            except:
                entropies[ch] = 0  # Default value if calculation fails
        
        return entropies
    
    def _calculate_fractal_dimension(self, data: np.ndarray) -> np.ndarray:
        """Calculate Higuchi fractal dimension for each channel"""
        fractal_dims = np.zeros(data.shape[0])
        
        for ch in range(data.shape[0]):
            signal_data = data[ch]
            N = len(signal_data)
            
            # Calculate for different k values
            k_max = min(10, N // 10)  # Reasonable maximum k
            k_values = range(1, k_max + 1)
            L_k = []
            
            for k in k_values:
                L_m = []
                for m in range(k):
                    L_km = 0
                    max_i = (N - m - 1) // k
                    if max_i > 0:
                        for i in range(1, max_i + 1):
                            L_km += abs(signal_data[m + i * k] - signal_data[m + (i - 1) * k])
                        L_km = L_km * (N - 1) / (max_i * k * k)
                        L_m.append(L_km)
                
                if L_m:
                    L_k.append(np.mean(L_m))
                else:
                    L_k.append(0)
            
            # Linear regression to find slope
            if len(L_k) > 1:
                log_k = np.log(k_values)
                log_L = np.log([l for l in L_k if l > 0])
                
                if len(log_L) > 1:
                    slope = np.polyfit(log_k[:len(log_L)], log_L, 1)[0]
                    fractal_dims[ch] = -slope
                else:
                    fractal_dims[ch] = 1.0  # Default value
            else:
                fractal_dims[ch] = 1.0  # Default value
        
        return fractal_dims
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize features according to configuration"""
        if self.config.normalization == 'z-score':
            return self._z_score_normalize(features)
        elif self.config.normalization == 'min-max':
            return self._min_max_normalize(features)
        else:
            return features
    
    def _z_score_normalize(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply z-score normalization to features"""
        normalized_features = features.copy()
        
        for category in ['time_domain', 'frequency_domain', 'advanced']:
            if category in features:
                for feature_name, feature_values in features[category].items():
                    if isinstance(feature_values, np.ndarray):
                        mean_val = np.mean(feature_values)
                        std_val = np.std(feature_values)
                        if std_val > 0:
                            normalized_features[category][feature_name] = (feature_values - mean_val) / std_val
        
        return normalized_features
    
    def _min_max_normalize(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Apply min-max normalization to features"""
        normalized_features = features.copy()
        
        for category in ['time_domain', 'frequency_domain', 'advanced']:
            if category in features:
                for feature_name, feature_values in features[category].items():
                    if isinstance(feature_values, np.ndarray):
                        min_val = np.min(feature_values)
                        max_val = np.max(feature_values)
                        if max_val > min_val:
                            normalized_features[category][feature_name] = (feature_values - min_val) / (max_val - min_val)
        
        return normalized_features
    
    def get_feature_vector(self, features: Dict[str, Any], selected_features: List[str] = None) -> np.ndarray:
        """
        Convert feature dictionary to a single feature vector
        
        Args:
            features: Feature dictionary from extract_features()
            selected_features: List of specific features to include
        
        Returns:
            1D numpy array containing concatenated features
        """
        feature_vector = []
        
        # Define default feature order
        if selected_features is None:
            selected_features = [
                'time_domain', 'frequency_domain', 'advanced'
            ]
        
        for category in selected_features:
            if category in features:
                for feature_name, feature_values in features[category].items():
                    if isinstance(feature_values, np.ndarray):
                        if feature_values.ndim == 1:
                            feature_vector.extend(feature_values)
                        elif feature_values.ndim == 2:
                            # For matrices, use upper triangular part
                            feature_vector.extend(feature_values[np.triu_indices(feature_values.shape[0])])
        
        return np.array(feature_vector)

def main():
    """Test the feature extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EEG Feature Extractor")
    parser.add_argument('--config', default='config/pipeline.yaml', help='Config file path')
    parser.add_argument('--channels', type=int, default=14, help='Number of EEG channels')
    parser.add_argument('--duration', type=float, default=4.0, help='Signal duration (seconds)')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = EEGFeatureExtractor(args.config)
    
    # Generate synthetic EEG data
    sampling_rate = extractor.config.sampling_rate
    n_samples = int(args.duration * sampling_rate)
    n_channels = args.channels
    
    print(f"Testing feature extraction on {n_channels} channels, {n_samples} samples ({args.duration}s)")
    
    # Create realistic synthetic EEG signal
    t = np.arange(n_samples) / sampling_rate
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Mix of frequency components
        alpha = 20 * np.sin(2 * np.pi * 10 * t + ch * 0.1)      # Alpha
        beta = 10 * np.sin(2 * np.pi * 20 * t + ch * 0.2)       # Beta
        theta = 15 * np.sin(2 * np.pi * 6 * t + ch * 0.15)      # Theta
        noise = np.random.normal(0, 5, n_samples)                # Noise
        
        data[ch] = alpha + beta + theta + noise
    
    # Extract features
    print("Extracting features...")
    features = extractor.extract_features(data)
    
    # Display results
    print(f"\nFeature extraction completed!")
    print(f"Metadata: {features['metadata']}")
    
    print(f"\nTime domain features:")
    for name, values in features['time_domain'].items():
        if isinstance(values, np.ndarray):
            print(f"  {name}: shape={values.shape}, mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    print(f"\nFrequency domain features:")
    for name, values in features['frequency_domain'].items():
        if isinstance(values, np.ndarray):
            print(f"  {name}: shape={values.shape}, mean={np.mean(values):.3f}, std={np.std(values):.3f}")
    
    if 'connectivity' in features:
        print(f"\nConnectivity features:")
        for name, values in features['connectivity'].items():
            if isinstance(values, np.ndarray):
                print(f"  {name}: shape={values.shape}, mean={np.mean(values):.3f}")
    
    print(f"\nAdvanced features:")
    for name, values in features['advanced'].items():
        if isinstance(values, np.ndarray):
            print(f"  {name}: shape={values.shape}, mean={np.mean(values):.3f}")
    
    # Create feature vector
    feature_vector = extractor.get_feature_vector(features)
    print(f"\nFeature vector: {len(feature_vector)} dimensions")
    
    print("Feature extraction test completed!")

if __name__ == "__main__":
    main()
