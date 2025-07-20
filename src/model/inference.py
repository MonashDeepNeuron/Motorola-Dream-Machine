#!/usr/bin/env python3
"""
EEG Model Inference Module
==========================

This module provides real-time inference capabilities using the trained EEG model
for brain-computer interface applications.
"""

import torch
import torch.nn as nn
import numpy as np
import yaml
import logging
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import joblib

# Import the model architecture
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from model.eeg_model import EEGClassifier
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: EEG model not available. Using mock inference.")

class EEGModelInference:
    """Real-time EEG model inference engine"""
    
    def __init__(self, config_path: str = "config/pipeline.yaml"):
        """Initialize model inference engine"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get model configuration
        self.model_config = self.config['model']
        self.inference_config = self.config.get('inference', {})
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Model state
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inference parameters
        self.confidence_threshold = self.inference_config.get('confidence_threshold', 0.7)
        self.prediction_smoothing = self.inference_config.get('prediction_smoothing', True)
        self.smoothing_window = self.inference_config.get('smoothing_window', 5)
        
        # Prediction history for smoothing
        self.prediction_history = []
        self.confidence_history = []
        
        # Statistics
        self.inference_count = 0
        self.total_inference_time = 0
        
        # Class mapping
        self.class_names = [
            'stop', 'move_x_pos', 'move_x_neg', 
            'move_y_pos', 'move_y_neg', 'move_z_pos', 'move_z_neg'
        ]
        
        self.logger.info(f"EEG inference engine initialized on device: {self.device}")
    
    def load_model(self, model_path: str = None, scaler_path: str = None) -> bool:
        """Load trained model and scaler"""
        if model_path is None:
            model_path = self.model_config.get('save_path', 'models/eeg_model.pth')
        
        if scaler_path is None:
            scaler_path = self.model_config.get('scaler_path', 'models/feature_scaler.pkl')
        
        try:
            if not MODEL_AVAILABLE:
                self.logger.warning("Model architecture not available - using mock inference")
                self.is_loaded = True
                return True
            
            # Load model architecture
            model_params = self.model_config['architecture']
            
            self.model = EEGClassifier(
                n_channels=model_params['n_channels'],
                n_classes=model_params['n_classes'],
                sampling_rate=model_params['sampling_rate'],
                cnn_filters=model_params['cnn_filters'],
                gcn_hidden=model_params['gcn_hidden'],
                transformer_dim=model_params['transformer_dim'],
                transformer_heads=model_params['transformer_heads'],
                transformer_layers=model_params['transformer_layers'],
                dropout=model_params['dropout']
            )
            
            # Load model weights
            if Path(model_path).exists():
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.logger.info(f"Loaded model from checkpoint: {model_path}")
                    else:
                        self.model.load_state_dict(checkpoint)
                        self.logger.info(f"Loaded model state dict: {model_path}")
                else:
                    # Legacy format
                    self.model = checkpoint
                    self.logger.info(f"Loaded complete model: {model_path}")
                
                self.model.to(self.device)
                self.model.eval()
                
            else:
                self.logger.warning(f"Model file not found: {model_path}. Using random weights.")
                self.model.to(self.device)
                self.model.eval()
            
            # Load feature scaler
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Loaded feature scaler: {scaler_path}")
            else:
                self.logger.warning(f"Scaler file not found: {scaler_path}. Features will not be scaled.")
            
            self.is_loaded = True
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make prediction from EEG features
        
        Args:
            features: Feature vector from EEG data
        
        Returns:
            Tuple of (prediction, confidence, details)
        """
        if not self.is_loaded:
            self.logger.error("Model not loaded")
            return 0, 0.0, {'error': 'Model not loaded'}
        
        start_time = time.time()
        
        try:
            # Use mock inference if model not available
            if not MODEL_AVAILABLE:
                return self._mock_prediction(features)
            
            # Prepare input features
            if self.scaler is not None:
                features = self.scaler.transform(features.reshape(1, -1))[0]
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                
                # Get prediction and confidence
                confidence, predicted = torch.max(probabilities, 1)
                prediction = predicted.item()
                confidence_score = confidence.item()
            
            # Apply smoothing if enabled
            if self.prediction_smoothing:
                prediction, confidence_score = self._apply_smoothing(prediction, confidence_score)
            
            # Calculate inference time
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            # Prepare detailed results
            details = {
                'class_name': self.class_names[prediction] if prediction < len(self.class_names) else 'unknown',
                'all_probabilities': probabilities.cpu().numpy()[0].tolist(),
                'inference_time': inference_time,
                'smoothed': self.prediction_smoothing,
                'meets_threshold': confidence_score >= self.confidence_threshold
            }
            
            return prediction, confidence_score, details
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return 0, 0.0, {'error': str(e)}
    
    def _mock_prediction(self, features: np.ndarray) -> Tuple[int, float, Dict[str, Any]]:
        """Mock prediction for testing when model is not available"""
        # Simple rule-based mock prediction based on feature statistics
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        
        # Use feature statistics to generate deterministic but varying predictions
        seed = int(abs(feature_mean * 1000)) % 1000
        np.random.seed(seed)
        
        prediction = np.random.randint(0, len(self.class_names))
        confidence = 0.5 + 0.4 * np.random.random()  # 0.5 to 0.9
        
        details = {
            'class_name': self.class_names[prediction],
            'all_probabilities': np.random.dirichlet(np.ones(len(self.class_names))).tolist(),
            'inference_time': 0.001,
            'smoothed': False,
            'meets_threshold': confidence >= self.confidence_threshold,
            'mock': True
        }
        
        return prediction, confidence, details
    
    def _apply_smoothing(self, prediction: int, confidence: float) -> Tuple[int, float]:
        """Apply temporal smoothing to predictions"""
        # Add to history
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.prediction_history) > self.smoothing_window:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
        
        # If not enough history, return original
        if len(self.prediction_history) < 3:
            return prediction, confidence
        
        # Count prediction occurrences
        prediction_counts = {}
        for pred in self.prediction_history:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # Use most frequent prediction
        smoothed_prediction = max(prediction_counts, key=prediction_counts.get)
        
        # Average confidence for the smoothed prediction
        smoothed_confidence = np.mean([
            conf for pred, conf in zip(self.prediction_history, self.confidence_history)
            if pred == smoothed_prediction
        ])
        
        return smoothed_prediction, float(smoothed_confidence)
    
    def predict_from_raw_eeg(self, eeg_data: np.ndarray, feature_extractor) -> Tuple[int, float, Dict[str, Any]]:
        """
        Make prediction directly from raw EEG data
        
        Args:
            eeg_data: Raw EEG data (channels x samples)
            feature_extractor: EEG feature extractor instance
        
        Returns:
            Tuple of (prediction, confidence, details)
        """
        try:
            # Extract features
            features_dict = feature_extractor.extract_features(eeg_data)
            
            # Convert to feature vector
            feature_vector = feature_extractor.get_feature_vector(features_dict)
            
            # Make prediction
            prediction, confidence, details = self.predict(feature_vector)
            
            # Add feature extraction info
            details['feature_extraction'] = {
                'n_features': len(feature_vector),
                'feature_range': (float(np.min(feature_vector)), float(np.max(feature_vector))),
                'feature_mean': float(np.mean(feature_vector))
            }
            
            return prediction, confidence, details
            
        except Exception as e:
            self.logger.error(f"Prediction from raw EEG failed: {e}")
            return 0, 0.0, {'error': str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get inference statistics"""
        avg_inference_time = (
            self.total_inference_time / self.inference_count 
            if self.inference_count > 0 else 0
        )
        
        return {
            'model_loaded': self.is_loaded,
            'device': str(self.device),
            'inference_count': self.inference_count,
            'average_inference_time': avg_inference_time,
            'total_inference_time': self.total_inference_time,
            'confidence_threshold': self.confidence_threshold,
            'prediction_smoothing': self.prediction_smoothing,
            'smoothing_window': self.smoothing_window,
            'history_length': len(self.prediction_history)
        }
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history.clear()
        self.confidence_history.clear()
        self.logger.info("Prediction history reset")
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for predictions"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        self.logger.info(f"Confidence threshold set to: {self.confidence_threshold}")
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        return self.class_names.copy()

def main():
    """Test the model inference engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test EEG Model Inference")
    parser.add_argument('--config', default='config/pipeline.yaml', help='Config file path')
    parser.add_argument('--model', help='Model file path')
    parser.add_argument('--scaler', help='Scaler file path')
    parser.add_argument('--iterations', type=int, default=100, help='Number of test iterations')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference = EEGModelInference(args.config)
    
    # Load model
    print("Loading model...")
    if not inference.load_model(args.model, args.scaler):
        print("Failed to load model - continuing with mock inference")
    
    print(f"Model statistics: {inference.get_statistics()}")
    print(f"Class names: {inference.get_class_names()}")
    
    # Test inference with synthetic features
    print(f"\nTesting inference with {args.iterations} synthetic feature vectors...")
    
    # Assume model expects certain number of features
    n_features = 100  # This should match your feature extractor output
    
    predictions_summary = {name: 0 for name in inference.get_class_names()}
    confidence_scores = []
    inference_times = []
    
    for i in range(args.iterations):
        # Generate synthetic features
        features = np.random.randn(n_features)
        
        # Make prediction
        prediction, confidence, details = inference.predict(features)
        
        # Record results
        class_name = details.get('class_name', 'unknown')
        predictions_summary[class_name] += 1
        confidence_scores.append(confidence)
        inference_times.append(details.get('inference_time', 0))
        
        if i < 10:  # Show first 10 predictions
            print(f"Prediction {i+1}: {class_name} (confidence: {confidence:.3f})")
    
    # Summary statistics
    print(f"\nInference Summary:")
    print(f"Total predictions: {args.iterations}")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"Average inference time: {np.mean(inference_times):.4f}s")
    print(f"Predictions per class: {predictions_summary}")
    
    # Final statistics
    final_stats = inference.get_statistics()
    print(f"\nFinal statistics: {final_stats}")
    
    print("Inference test completed!")

if __name__ == "__main__":
    main()
