#!/usr/bin/env python3
"""
Preprocess Training Data
========================

Clean, normalize, and prepare EEG data for model training.

Usage:
    python3 training/preprocess_data.py --input <data_file> --output <processed_file>

Output:
    - Cleaned and normalized EEG data
    - Feature extraction and selection
    - Train/validation/test splits
    - Data quality reports
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.eeg.features import EEGFeatureExtractor
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class EEGDataPreprocessor:
    """Preprocesses EEG data for machine learning."""
    
    def __init__(self, sampling_rate=256):
        self.sampling_rate = sampling_rate
        self.feature_extractor = EEGFeatureExtractor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        self.preprocessing_stats = {
            'original_samples': 0,
            'filtered_samples': 0,
            'feature_dimension': 0,
            'class_distribution': {},
            'quality_metrics': {}
        }
    
    def load_training_data(self, data_file):
        """Load training data from file."""
        print(f"📁 Loading training data from: {data_file}")
        
        data_file = Path(data_file)
        
        if data_file.suffix == '.jsonl':
            # Load JSONL format
            samples = []
            with open(data_file, 'r') as f:
                for line in f:
                    samples.append(json.loads(line))
            
            # Extract EEG data and labels
            eeg_data = [np.array(sample['eeg_data']) for sample in samples]
            labels = [sample['command'] for sample in samples]
            
        elif data_file.suffix == '.json':
            # Load JSON format
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            eeg_data = [np.array(sample) for sample in data['samples']]
            labels = data['labels']
            
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")
        
        print(f"   ✅ Loaded {len(eeg_data)} samples")
        print(f"   📊 Data shape: {eeg_data[0].shape if eeg_data else 'Empty'}")
        
        self.preprocessing_stats['original_samples'] = len(eeg_data)
        
        return eeg_data, labels
    
    def analyze_data_quality(self, eeg_data, labels):
        """Analyze the quality of the input data."""
        print(f"\n🔍 Analyzing data quality...")
        
        # Basic statistics
        n_samples = len(eeg_data)
        n_channels, n_timepoints = eeg_data[0].shape if eeg_data else (0, 0)
        
        print(f"   📈 Dataset statistics:")
        print(f"     Samples: {n_samples}")
        print(f"     Channels: {n_channels}")
        print(f"     Time points: {n_timepoints}")
        print(f"     Duration per sample: {n_timepoints/self.sampling_rate:.1f}s")
        
        # Class distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique_labels, counts))
        
        print(f"\n   📋 Class distribution:")
        for label, count in class_dist.items():
            percentage = count / n_samples * 100
            print(f"     {label:15s}: {count:3d} samples ({percentage:5.1f}%)")
        
        # Data quality checks
        print(f"\n   🔍 Quality checks:")
        
        # Check for consistent shapes
        shapes = [data.shape for data in eeg_data]
        consistent_shapes = len(set(shapes)) == 1
        print(f"     Consistent shapes: {'✅' if consistent_shapes else '❌'}")
        
        # Check for missing values
        has_nan = any(np.isnan(data).any() for data in eeg_data)
        print(f"     No missing values: {'✅' if not has_nan else '❌'}")
        
        # Check signal amplitudes
        amplitudes = [np.std(data) for data in eeg_data]
        avg_amplitude = np.mean(amplitudes)
        std_amplitude = np.std(amplitudes)
        
        print(f"     Average signal std: {avg_amplitude:.3f}")
        print(f"     Amplitude consistency: {std_amplitude:.3f}")
        print(f"     Signal quality: {'✅ Good' if 0.1 < avg_amplitude < 10 else '⚠️  Check'}")
        
        # Store quality metrics
        self.preprocessing_stats['quality_metrics'] = {
            'consistent_shapes': consistent_shapes,
            'has_missing_values': has_nan,
            'average_amplitude': avg_amplitude,
            'amplitude_std': std_amplitude,
            'signal_quality_ok': 0.1 < avg_amplitude < 10
        }
        
        self.preprocessing_stats['class_distribution'] = class_dist
        
        return consistent_shapes and not has_nan
    
    def filter_and_clean(self, eeg_data, labels):
        """Filter and clean the EEG data."""
        print(f"\n🧹 Filtering and cleaning data...")
        
        filtered_data = []
        filtered_labels = []
        
        removed_count = 0
        
        for i, (data, label) in enumerate(zip(eeg_data, labels)):
            # Check for artifacts
            keep_sample = True
            
            # Remove samples with extreme amplitudes (likely artifacts)
            if np.max(np.abs(data)) > 100:  # Adjust threshold as needed
                keep_sample = False
                removed_count += 1
            
            # Remove samples with flat signals (equipment issues)
            if np.std(data) < 0.01:
                keep_sample = False
                removed_count += 1
            
            # Remove samples with NaN or inf values
            if np.isnan(data).any() or np.isinf(data).any():
                keep_sample = False
                removed_count += 1
            
            if keep_sample:
                filtered_data.append(data)
                filtered_labels.append(label)
        
        print(f"   🗑️  Removed {removed_count} poor quality samples")
        print(f"   ✅ Kept {len(filtered_data)} good quality samples")
        
        self.preprocessing_stats['filtered_samples'] = len(filtered_data)
        
        return filtered_data, filtered_labels
    
    def extract_features(self, eeg_data):
        """Extract features from EEG data."""
        print(f"\n🔬 Extracting features...")
        
        features = []
        
        for i, data in enumerate(eeg_data):
            if i % 10 == 0:
                print(f"   Processing sample {i+1}/{len(eeg_data)}")
            
            # Extract features using the feature extractor
            sample_features = self.feature_extractor.extract_features(data)
            features.append(sample_features)
        
        features = np.array(features)
        
        print(f"   ✅ Extracted features: {features.shape}")
        print(f"   📊 Feature dimension: {features.shape[1]}")
        
        self.preprocessing_stats['feature_dimension'] = features.shape[1]
        
        return features
    
    def normalize_features(self, features):
        """Normalize features using StandardScaler."""
        print(f"\n📏 Normalizing features...")
        
        # Fit and transform the features
        normalized_features = self.scaler.fit_transform(features)
        
        print(f"   ✅ Features normalized")
        print(f"   📊 Feature mean: {np.mean(normalized_features, axis=0)[:5]}... (first 5)")
        print(f"   📊 Feature std: {np.std(normalized_features, axis=0)[:5]}... (first 5)")
        
        return normalized_features
    
    def encode_labels(self, labels):
        """Encode labels to integers."""
        print(f"\n🏷️  Encoding labels...")
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"   ✅ Labels encoded")
        print(f"   📋 Label mapping:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"     {label:15s}: {i}")
        
        return encoded_labels
    
    def create_data_splits(self, features, labels, test_size=0.2, val_size=0.1):
        """Create train/validation/test splits."""
        print(f"\n✂️  Creating data splits...")
        print(f"   Test size: {test_size:.1%}")
        print(f"   Validation size: {val_size:.1%}")
        print(f"   Train size: {1-test_size-val_size:.1%}")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, 
            test_size=test_size, 
            stratify=labels, 
            random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=42
        )
        
        print(f"   ✅ Data splits created:")
        print(f"     Train: {X_train.shape[0]} samples")
        print(f"     Validation: {X_val.shape[0]} samples") 
        print(f"     Test: {X_test.shape[0]} samples")
        
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    
    def create_visualizations(self, features, labels, output_dir):
        """Create visualizations of the preprocessed data."""
        print(f"\n📊 Creating visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('EEG Data Preprocessing Analysis', fontsize=16)
        
        # 1. Class distribution
        ax1 = axes[0, 0]
        unique_labels, counts = np.unique(labels, return_counts=True)
        bars = ax1.bar(range(len(unique_labels)), counts)
        ax1.set_title('Class Distribution')
        ax1.set_xlabel('Command Classes')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(unique_labels)))
        ax1.set_xticklabels([str(label).replace('_', '\n') for label in unique_labels], rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{count}', ha='center', va='bottom')
        
        # 2. Feature distribution (first 20 features)
        ax2 = axes[0, 1]
        n_features_to_show = min(20, features.shape[1])
        feature_means = np.mean(features[:, :n_features_to_show], axis=0)
        feature_stds = np.std(features[:, :n_features_to_show], axis=0)
        
        x_pos = np.arange(n_features_to_show)
        ax2.bar(x_pos, feature_means, yerr=feature_stds, capsize=3)
        ax2.set_title(f'Feature Statistics (First {n_features_to_show})')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Feature Value')
        
        # 3. Feature correlation matrix (subset)
        ax3 = axes[1, 0]
        n_features_corr = min(50, features.shape[1])
        corr_matrix = np.corrcoef(features[:, :n_features_corr].T)
        
        im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title(f'Feature Correlations (First {n_features_corr})')
        ax3.set_xlabel('Feature Index')
        ax3.set_ylabel('Feature Index')
        plt.colorbar(im, ax=ax3)
        
        # 4. Sample feature vectors by class
        ax4 = axes[1, 1]
        
        # Plot mean feature vector for each class
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.any(mask):
                class_features = features[mask]
                mean_features = np.mean(class_features, axis=0)
                
                # Plot first 50 features
                n_plot = min(50, len(mean_features))
                ax4.plot(range(n_plot), mean_features[:n_plot], 
                        label=str(label), alpha=0.7, linewidth=2)
        
        ax4.set_title('Mean Feature Vectors by Class')
        ax4.set_xlabel('Feature Index')
        ax4.set_ylabel('Feature Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'preprocessing_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   💾 Visualization saved to: {output_dir / 'preprocessing_analysis.png'}")
    
    def save_preprocessed_data(self, data_splits, label_splits, output_file):
        """Save the preprocessed data."""
        print(f"\n💾 Saving preprocessed data...")
        
        X_train, X_val, X_test = data_splits
        y_train, y_val, y_test = label_splits
        
        preprocessed_data = {
            'X_train': X_train.tolist(),
            'X_val': X_val.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_val': y_val.tolist(),
            'y_test': y_test.tolist(),
            'label_classes': self.label_encoder.classes_.tolist(),
            'feature_scaler_mean': self.scaler.mean_.tolist(),
            'feature_scaler_scale': self.scaler.scale_.tolist(),
            'preprocessing_stats': self.preprocessing_stats,
            'data_info': {
                'n_classes': len(self.label_encoder.classes_),
                'n_features': X_train.shape[1],
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0]
            }
        }
        
        output_file = Path(output_file)
        with open(output_file, 'w') as f:
            json.dump(preprocessed_data, f, indent=2)
        
        print(f"   ✅ Preprocessed data saved to: {output_file}")
        print(f"   📊 Ready for model training")
        
        return str(output_file)

def main():
    """Main preprocessing interface."""
    parser = argparse.ArgumentParser(description="Preprocess EEG training data")
    parser.add_argument('--input', required=True, help='Input data file')
    parser.add_argument('--output', help='Output preprocessed file')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Setup output file
    if not args.output:
        input_path = Path(args.input)
        args.output = input_path.parent / f"{input_path.stem}_preprocessed.json"
    
    print("🧹 EEG Data Preprocessing")
    print("=" * 40)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    
    # Initialize preprocessor
    preprocessor = EEGDataPreprocessor()
    
    try:
        # Load data
        eeg_data, labels = preprocessor.load_training_data(args.input)
        
        # Analyze quality
        quality_ok = preprocessor.analyze_data_quality(eeg_data, labels)
        
        if not quality_ok:
            print("⚠️  Data quality issues detected. Proceeding with caution...")
        
        # Filter and clean
        eeg_data, labels = preprocessor.filter_and_clean(eeg_data, labels)
        
        # Extract features
        features = preprocessor.extract_features(eeg_data)
        
        # Normalize features
        features = preprocessor.normalize_features(features)
        
        # Encode labels
        labels = preprocessor.encode_labels(labels)
        
        # Create data splits
        data_splits, label_splits = preprocessor.create_data_splits(
            features, labels, args.test_size, args.val_size
        )
        
        # Create visualizations
        if args.visualize:
            output_dir = Path(args.output).parent / "preprocessing_output"
            preprocessor.create_visualizations(features, labels, output_dir)
        
        # Save preprocessed data
        output_file = preprocessor.save_preprocessed_data(data_splits, label_splits, args.output)
        
        print(f"\n✅ Preprocessing completed!")
        print(f"📁 Preprocessed data: {output_file}")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review preprocessing statistics")
        print(f"   2. Run: python3 training/train_model.py --data {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
