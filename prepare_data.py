#!/usr/bin/env python3
"""
EEG Data Preparation Script
===========================

This script processes EEG files and creates training/testing datasets
for the unified pipeline.
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project paths
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(current_file)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "eeg_pipeline"))

import mne
from eeg_pipeline.analysis.bands import compute_window_band_power

class EEGDataPreprocessor:
    """Preprocesses EEG data for training/testing"""
    
    def __init__(self, config_path: str = "pipeline_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.sample_rate = self.config["pipeline_config"]["sample_rate"]
        self.window_size = self.config["pipeline_config"]["window_size"]
        self.step_size = self.config["pipeline_config"]["step_size"]
        self.freq_bands = {k: tuple(v) for k, v in self.config["frequency_bands"].items()}
    
    def process_edf_file(self, edf_path: str) -> Tuple[np.ndarray, List[str], Dict]:
        """Process a single EDF file"""
        print(f"Processing: {edf_path}")
        
        # Load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
        raw.rename_channels(lambda x: x.strip('. ').replace('..', '').upper())
        
        # Get basic info
        original_sfreq = raw.info['sfreq']
        channel_names = raw.ch_names
        
        # Resample if needed
        if original_sfreq != self.sample_rate:
            print(f"  Resampling from {original_sfreq} Hz to {self.sample_rate} Hz")
            raw.resample(self.sample_rate)
        
        # Get data in volts
        data_volts = raw.get_data(units='V')
        
        # Extract annotations/events for labeling
        annotations = self.extract_annotations(raw)
        
        # Create sliding windows and extract features
        features, labels = self.create_windowed_features(data_volts, annotations)
        
        metadata = {
            "file_path": edf_path,
            "original_sfreq": original_sfreq,
            "resampled_sfreq": self.sample_rate,
            "n_channels": len(channel_names),
            "channel_names": channel_names,
            "n_windows": len(features),
            "annotations": annotations
        }
        
        return features, labels, metadata
    
    def extract_annotations(self, raw: mne.io.BaseRaw) -> Dict:
        """Extract event annotations for labeling"""
        annotations = {
            "onset": [],
            "duration": [],
            "description": []
        }
        
        if hasattr(raw, 'annotations') and len(raw.annotations) > 0:
            annotations["onset"] = raw.annotations.onset.tolist()
            annotations["duration"] = raw.annotations.duration.tolist()
            annotations["description"] = raw.annotations.description.tolist()
        
        return annotations
    
    def create_windowed_features(self, data_volts: np.ndarray, annotations: Dict) -> Tuple[List[np.ndarray], List[int]]:
        """Create sliding windows and extract frequency domain features"""
        n_channels, n_samples = data_volts.shape
        
        window_samples = int(self.window_size * self.sample_rate)
        step_samples = int(self.step_size * self.sample_rate)
        
        features = []
        labels = []
        
        # Create MNE info for band power computation
        ch_names = [f"CH{i:02d}" for i in range(n_channels)]
        info = mne.create_info(ch_names=ch_names, sfreq=self.sample_rate, ch_types='eeg')
        
        for start_idx in range(0, n_samples - window_samples + 1, step_samples):
            end_idx = start_idx + window_samples
            window_data = data_volts[:, start_idx:end_idx]
            
            # Create temporary Raw object for this window
            raw_window = mne.io.RawArray(window_data, info, verbose='ERROR')
            
            # Compute band power features
            try:
                window_results, _ = compute_window_band_power(
                    raw_window,
                    window_size_s=self.window_size,
                    step_size_s=self.window_size,  # No overlap for single window
                    bands=self.freq_bands
                )
                
                # Organize features: [n_channels, n_bands]
                window_features = np.zeros((n_channels, len(self.freq_bands)))
                
                for result in window_results:
                    ch_idx = int(result.channel_label.replace("CH", ""))
                    band_names = list(self.freq_bands.keys())
                    if result.band_name in band_names:
                        band_idx = band_names.index(result.band_name)
                        window_features[ch_idx, band_idx] = result.power
                
                features.append(window_features)
                
                # Determine label based on annotations
                window_center_time = (start_idx + window_samples // 2) / self.sample_rate
                label = self.get_label_for_time(window_center_time, annotations)
                labels.append(label)
                
            except Exception as e:
                print(f"  Warning: Failed to process window at {start_idx}: {e}")
                continue
        
        return features, labels
    
    def get_label_for_time(self, time_sec: float, annotations: Dict) -> int:
        """Determine label for a given time based on annotations"""
        # Simple labeling scheme - you can modify this based on your annotation format
        
        if not annotations["onset"]:
            return 0  # Default: rest/baseline
        
        # Check if time falls within any annotation
        for i, (onset, duration, description) in enumerate(zip(
            annotations["onset"], annotations["duration"], annotations["description"]
        )):
            if onset <= time_sec <= onset + duration:
                # Map description to class label
                desc_lower = description.lower()
                if 'left' in desc_lower or 't1' in desc_lower:
                    return 1
                elif 'right' in desc_lower or 't2' in desc_lower:
                    return 2
                elif 'forward' in desc_lower or 'up' in desc_lower:
                    return 3
                elif 'backward' in desc_lower or 'down' in desc_lower:
                    return 4
                else:
                    return 0  # Rest
        
        return 0  # Default: rest
    
    def process_multiple_files(self, edf_files: List[str], output_dir: str = "data/processed"):
        """Process multiple EDF files and save as training data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_features = []
        all_labels = []
        all_metadata = []
        
        for edf_file in edf_files:
            try:
                features, labels, metadata = self.process_edf_file(edf_file)
                all_features.extend(features)
                all_labels.extend(labels)
                all_metadata.append(metadata)
                
                print(f"  Extracted {len(features)} windows with labels: {set(labels)}")
                
            except Exception as e:
                print(f"Error processing {edf_file}: {e}")
                continue
        
        # Convert to numpy arrays
        features_array = np.array(all_features)  # Shape: (n_windows, n_channels, n_bands)
        labels_array = np.array(all_labels)      # Shape: (n_windows,)
        
        # Save processed data
        np.save(output_path / "features.npy", features_array)
        np.save(output_path / "labels.npy", labels_array)
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump({
                "processing_config": self.config,
                "total_windows": len(features_array),
                "feature_shape": list(features_array.shape),
                "label_distribution": {int(k): int(v) for k, v in zip(*np.unique(labels_array, return_counts=True))},
                "file_metadata": all_metadata
            }, f, indent=2)
        
        print(f"\nProcessed data saved to {output_path}")
        print(f"Features shape: {features_array.shape}")
        print(f"Labels shape: {labels_array.shape}")
        print(f"Label distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")
        
        return features_array, labels_array

def main():
    parser = argparse.ArgumentParser(description="EEG Data Preprocessor")
    parser.add_argument('--edf-files', nargs='+', required=True,
                       help='EDF files to process')
    parser.add_argument('--output-dir', default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--config', default='pipeline_config.json',
                       help='Configuration file')
    
    args = parser.parse_args()
    
    # Expand glob patterns if needed
    edf_files = []
    for pattern in args.edf_files:
        if '*' in pattern:
            from glob import glob
            edf_files.extend(glob(pattern))
        else:
            edf_files.append(pattern)
    
    # Filter to existing files
    edf_files = [f for f in edf_files if os.path.exists(f)]
    
    if not edf_files:
        print("No valid EDF files found!")
        return
    
    print(f"Processing {len(edf_files)} EDF files...")
    
    # Process files
    preprocessor = EEGDataPreprocessor(args.config)
    preprocessor.process_multiple_files(edf_files, args.output_dir)

if __name__ == "__main__":
    main()
