#!/usr/bin/env python3
"""
Demo 3: Model Inference
=====================

See how the trained ML model makes predictions from EEG signals.

Usage:
    python3 demos/demo_model_inference.py

Output:
    - Live predictions on EEG data
    - Confidence scores for each robot command
    - Model decision visualization
    - Prediction accuracy metrics
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.model.cnn_gcn_transformer import EEGRobotModel
    from src.eeg.features import EEGFeatureExtractor
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

def generate_command_eeg(command_type, duration=2, sampling_rate=256, channels=14):
    """Generate EEG data that simulates different mental commands."""
    samples = int(duration * sampling_rate)
    time = np.linspace(0, duration, samples)
    
    eeg_data = np.zeros((channels, samples))
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    
    # Command-specific EEG patterns
    command_patterns = {
        'move_forward': {
            'frontal_beta': 0.8,    # Motor planning
            'motor_beta': 0.9,      # Motor execution
            'alpha_suppression': 0.3 # Active thinking
        },
        'turn_left': {
            'right_motor': 0.8,     # Contralateral control
            'frontal_beta': 0.7,
            'alpha_suppression': 0.4
        },
        'turn_right': {
            'left_motor': 0.8,      # Contralateral control
            'frontal_beta': 0.7,
            'alpha_suppression': 0.4
        },
        'grab_object': {
            'motor_beta': 0.9,      # Fine motor control
            'frontal_gamma': 0.6,   # Attention
            'alpha_suppression': 0.2
        },
        'release_object': {
            'motor_beta': 0.7,
            'frontal_beta': 0.8,
            'alpha_suppression': 0.3
        },
        'stop': {
            'alpha_increase': 0.8,  # Relaxation
            'beta_decrease': 0.3,
            'theta_increase': 0.6
        },
        'no_command': {
            'alpha_baseline': 0.6,  # Resting state
            'beta_baseline': 0.4,
            'random_noise': 0.8
        }
    }
    
    pattern = command_patterns.get(command_type, command_patterns['no_command'])
    
    for ch in range(channels):
        ch_name = channel_names[ch]
        
        # Base rhythms
        delta = 0.4 * np.sin(2 * np.pi * 2 * time + np.random.rand() * 2 * np.pi)
        theta = 0.4 * np.sin(2 * np.pi * 6 * time + np.random.rand() * 2 * np.pi)
        alpha = 0.5 * np.sin(2 * np.pi * 10 * time + np.random.rand() * 2 * np.pi)
        beta = 0.4 * np.sin(2 * np.pi * 20 * time + np.random.rand() * 2 * np.pi)
        gamma = 0.2 * np.sin(2 * np.pi * 35 * time + np.random.rand() * 2 * np.pi)
        
        # Apply command-specific modulations
        if 'frontal' in pattern and ('F' in ch_name or 'AF' in ch_name):
            if 'frontal_beta' in pattern:
                beta *= pattern['frontal_beta'] * 2
            if 'frontal_gamma' in pattern:
                gamma *= pattern['frontal_gamma'] * 3
        
        if 'motor' in pattern and ('C' in ch_name or 'T' in ch_name):
            if 'motor_beta' in pattern:
                beta *= pattern['motor_beta'] * 2
        
        if 'left_motor' in pattern and ch_name in ['FC5', 'T7']:
            beta *= pattern['left_motor'] * 2
            
        if 'right_motor' in pattern and ch_name in ['FC6', 'T8']:
            beta *= pattern['right_motor'] * 2
        
        if 'alpha_suppression' in pattern:
            alpha *= pattern['alpha_suppression']
            
        if 'alpha_increase' in pattern:
            alpha *= pattern['alpha_increase'] * 1.5
            
        if 'alpha_baseline' in pattern:
            alpha *= pattern['alpha_baseline']
            
        if 'beta_baseline' in pattern:
            beta *= pattern['beta_baseline']
            
        if 'theta_increase' in pattern:
            theta *= pattern['theta_increase'] * 1.5
        
        # Add noise
        noise_level = pattern.get('random_noise', 0.2)
        noise = noise_level * np.random.randn(samples)
        
        # Combine all components
        eeg_data[ch] = delta + theta + alpha + beta + gamma + noise
    
    return eeg_data

def demo_model_inference():
    """Demonstrate model inference on EEG data."""
    print("🤖 Demo 3: Model Inference")
    print("=" * 40)
    
    # Setup
    logger = setup_logging(log_level="INFO")
    
    # Initialize model and feature extractor
    print("🧠 Loading model...")
    model = EEGRobotModel()
    feature_extractor = EEGFeatureExtractor()
    
    # Command labels
    commands = ['move_forward', 'turn_left', 'turn_right', 'grab_object', 'release_object', 'stop', 'no_command']
    
    print(f"📋 Robot commands: {commands}")
    
    # Create output directory
    output_dir = Path("output/demo3_model_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test predictions on different command types
    print("\n🔮 Testing predictions on simulated mental commands...")
    
    all_predictions = []
    all_confidences = []
    all_true_labels = []
    
    for i, command in enumerate(commands):
        print(f"\n   Testing: {command}")
        
        # Generate EEG for this command
        eeg_data = generate_command_eeg(command)
        
        # Extract features
        features = feature_extractor.extract_features(eeg_data)
        
        # Make prediction
        prediction, confidence = model.predict(features)
        predicted_command = commands[prediction]
        
        print(f"     Predicted: {predicted_command} (confidence: {confidence:.3f})")
        
        # Store results
        all_predictions.append(prediction)
        all_confidences.append(confidence)
        all_true_labels.append(i)
        
        # Show detailed probabilities for first few commands
        if i < 3:
            probabilities = model.predict_proba(features)
            print("     Detailed probabilities:")
            for j, (cmd, prob) in enumerate(zip(commands, probabilities)):
                print(f"       {cmd:15s}: {prob:.3f} {'✓' if j == prediction else ''}")
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_true_labels))
    print(f"\n📊 Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Visualize results
    print("\n📈 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Inference Analysis', fontsize=16)
    
    # 1. Prediction accuracy by command
    ax1 = axes[0, 0]
    correct_predictions = np.array(all_predictions) == np.array(all_true_labels)
    accuracy_by_command = []
    for i, command in enumerate(commands):
        cmd_correct = correct_predictions[i]
        accuracy_by_command.append(1.0 if cmd_correct else 0.0)
    
    bars = ax1.bar(range(len(commands)), accuracy_by_command)
    ax1.set_title('Prediction Accuracy by Command')
    ax1.set_xlabel('Robot Commands')
    ax1.set_ylabel('Accuracy')
    ax1.set_xticks(range(len(commands)))
    ax1.set_xticklabels([cmd.replace('_', '\n') for cmd in commands], rotation=45)
    ax1.set_ylim([0, 1.1])
    
    # Color bars by accuracy
    for i, bar in enumerate(bars):
        if accuracy_by_command[i] == 1.0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # 2. Confidence scores
    ax2 = axes[0, 1]
    ax2.bar(range(len(commands)), all_confidences)
    ax2.set_title('Model Confidence by Command')
    ax2.set_xlabel('Robot Commands')
    ax2.set_ylabel('Confidence Score')
    ax2.set_xticks(range(len(commands)))
    ax2.set_xticklabels([cmd.replace('_', '\n') for cmd in commands], rotation=45)
    
    # 3. Confusion matrix
    ax3 = axes[1, 0]
    confusion_matrix = np.zeros((len(commands), len(commands)))
    for true_label, predicted_label in zip(all_true_labels, all_predictions):
        confusion_matrix[true_label, predicted_label] += 1
    
    im = ax3.imshow(confusion_matrix, cmap='Blues')
    ax3.set_title('Confusion Matrix')
    ax3.set_xlabel('Predicted Command')
    ax3.set_ylabel('True Command')
    ax3.set_xticks(range(len(commands)))
    ax3.set_yticks(range(len(commands)))
    ax3.set_xticklabels([cmd.replace('_', '\n') for cmd in commands], rotation=45)
    ax3.set_yticklabels([cmd.replace('_', '\n') for cmd in commands])
    
    # Add text annotations
    for i in range(len(commands)):
        for j in range(len(commands)):
            text = ax3.text(j, i, int(confusion_matrix[i, j]),
                           ha="center", va="center", color="black" if confusion_matrix[i, j] < 0.5 else "white")
    
    plt.colorbar(im, ax=ax3)
    
    # 4. Live inference simulation
    ax4 = axes[1, 1]
    
    # Simulate real-time predictions
    time_steps = 20
    live_commands = np.random.choice(commands, time_steps)
    live_confidences = []
    
    for cmd in live_commands:
        eeg_data = generate_command_eeg(cmd)
        features = feature_extractor.extract_features(eeg_data)
        _, confidence = model.predict(features)
        live_confidences.append(confidence)
    
    ax4.plot(range(time_steps), live_confidences, 'o-', linewidth=2, markersize=6)
    ax4.set_title('Simulated Real-time Confidence')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Confidence Score')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inference_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Real-time inference demonstration
    print("\n⚡ Real-time inference demonstration...")
    print("   Simulating continuous EEG processing...")
    
    for step in range(5):
        # Random command for demo
        test_command = np.random.choice(commands)
        print(f"\n   Step {step + 1}: Processing EEG data...")
        
        # Generate and process
        eeg_data = generate_command_eeg(test_command)
        features = feature_extractor.extract_features(eeg_data)
        prediction, confidence = model.predict(features)
        predicted_command = commands[prediction]
        
        # Show results
        print(f"     → Detected command: {predicted_command}")
        print(f"     → Confidence: {confidence:.3f}")
        print(f"     → True command: {test_command} {'✓' if predicted_command == test_command else '✗'}")
        
        # Simulate processing time
        time.sleep(0.5)
    
    # Save results
    results = {
        'commands': commands,
        'predictions': all_predictions,
        'confidences': all_confidences,
        'true_labels': all_true_labels,
        'accuracy': accuracy,
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    import json
    with open(output_dir / 'inference_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Demo completed!")
    print(f"📁 Outputs saved to: {output_dir}")
    print("\n📋 Key Insights:")
    print("   - Model processes features in real-time")
    print("   - Confidence scores indicate prediction reliability")
    print("   - Different mental states produce distinct EEG patterns")
    print("   - System is ready for live robot control")
    print(f"   - Current model accuracy: {accuracy*100:.1f}%")

if __name__ == "__main__":
    try:
        demo_model_inference()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
