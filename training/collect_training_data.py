#!/usr/bin/env python3
"""
Collect Training Data
====================

Collect and label EEG data for training the robot control model.

Usage:
    python3 training/collect_training_data.py --session <session_name> --duration <minutes>

Output:
    - Labeled EEG datasets
    - Training data in JSONL format
    - Session metadata and quality metrics
"""

import argparse
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.eeg.realtime import RealtimeEEGProcessor
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class TrainingDataCollector:
    """Collects labeled EEG data for training."""
    
    def __init__(self, session_name="training_session", output_dir="data/training"):
        self.session_name = session_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_file = self.output_dir / f"{session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.metadata_file = self.output_dir / f"{session_name}_metadata.json"
        
        self.commands = [
            'move_forward',
            'turn_left', 
            'turn_right',
            'grab_object',
            'release_object',
            'stop',
            'no_command'
        ]
        
        self.collected_data = []
        self.session_metadata = {
            'session_name': session_name,
            'start_time': datetime.now().isoformat(),
            'commands': self.commands,
            'samples_per_command': {},
            'total_samples': 0
        }
        
        # Initialize EEG processor
        try:
            self.eeg_processor = RealtimeEEGProcessor()
            self.simulation_mode = False
            print("✅ EEG hardware connected")
        except Exception as e:
            print(f"⚠️  EEG hardware not available: {e}")
            print("🔄 Using simulation mode")
            self.simulation_mode = True
    
    def generate_simulated_eeg(self, command, duration=4):
        """Generate realistic EEG data for a given command."""
        sampling_rate = 256
        samples = int(duration * sampling_rate)
        channels = 14
        time = np.linspace(0, duration, samples)
        
        eeg_data = np.zeros((channels, samples))
        
        # Command-specific patterns (same as in demos)
        command_patterns = {
            'move_forward': {'frontal_beta': 0.8, 'motor_beta': 0.9, 'alpha_suppression': 0.3},
            'turn_left': {'right_motor': 0.8, 'frontal_beta': 0.7, 'alpha_suppression': 0.4},
            'turn_right': {'left_motor': 0.8, 'frontal_beta': 0.7, 'alpha_suppression': 0.4},
            'grab_object': {'motor_beta': 0.9, 'frontal_gamma': 0.6, 'alpha_suppression': 0.2},
            'release_object': {'motor_beta': 0.7, 'frontal_beta': 0.8, 'alpha_suppression': 0.3},
            'stop': {'alpha_increase': 0.8, 'beta_decrease': 0.3, 'theta_increase': 0.6},
            'no_command': {'alpha_baseline': 0.6, 'beta_baseline': 0.4, 'random_noise': 0.8}
        }
        
        pattern = command_patterns.get(command, command_patterns['no_command'])
        channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        
        for ch in range(channels):
            ch_name = channel_names[ch]
            
            # Generate base rhythms
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
            
            # Apply other modulations
            for key, value in pattern.items():
                if key == 'alpha_suppression':
                    alpha *= value
                elif key == 'alpha_increase':
                    alpha *= value * 1.5
                elif key == 'alpha_baseline':
                    alpha *= value
                elif key == 'beta_baseline':
                    beta *= value
                elif key == 'theta_increase':
                    theta *= value * 1.5
            
            # Add noise
            noise_level = pattern.get('random_noise', 0.2)
            noise = noise_level * np.random.randn(samples)
            
            # Combine components
            eeg_data[ch] = delta + theta + alpha + beta + gamma + noise
        
        return eeg_data
    
    def collect_command_data(self, command, num_samples=5, trial_duration=4):
        """Collect data for a specific command."""
        print(f"\n🎯 Collecting data for command: {command}")
        print(f"   Collecting {num_samples} samples, {trial_duration}s each")
        
        command_data = []
        
        for trial in range(num_samples):
            print(f"\n   Trial {trial + 1}/{num_samples}")
            
            if not self.simulation_mode:
                # Real EEG collection
                input(f"   Press Enter when ready to think '{command}' for {trial_duration}s...")
                print("   🔴 Recording... Think about the command!")
                
                # Record EEG data
                eeg_data = self.eeg_processor.collect_data(duration=trial_duration)
                
                print("   ✅ Recording complete")
                
            else:
                # Simulated collection
                input(f"   [SIMULATION] Press Enter to simulate thinking '{command}'...")
                print("   🔴 [SIMULATION] Recording...")
                
                # Simulate recording time
                time.sleep(1)
                
                # Generate simulated data
                eeg_data = self.generate_simulated_eeg(command, trial_duration)
                
                print("   ✅ [SIMULATION] Recording complete")
            
            # Create data sample
            sample = {
                'timestamp': datetime.now().isoformat(),
                'command': command,
                'trial': trial + 1,
                'duration': trial_duration,
                'eeg_data': eeg_data.tolist(),
                'data_shape': list(eeg_data.shape),
                'sampling_rate': 256,
                'channels': 14,
                'simulation': self.simulation_mode
            }
            
            command_data.append(sample)
            self.collected_data.append(sample)
            
            # Save sample immediately (streaming save)
            with open(self.session_file, 'a') as f:
                f.write(json.dumps(sample) + '\n')
            
            print(f"   💾 Sample saved")
            
            # Brief pause between trials
            if trial < num_samples - 1:
                print(f"   ⏳ Rest for 3 seconds...")
                time.sleep(3)
        
        # Update metadata
        self.session_metadata['samples_per_command'][command] = num_samples
        self.session_metadata['total_samples'] += num_samples
        
        print(f"   ✅ Completed {num_samples} samples for '{command}'")
        return command_data
    
    def collect_balanced_dataset(self, samples_per_command=5):
        """Collect a balanced dataset across all commands."""
        print(f"🗂️  Collecting balanced dataset")
        print(f"   Commands: {self.commands}")
        print(f"   Samples per command: {samples_per_command}")
        print(f"   Total samples: {len(self.commands) * samples_per_command}")
        
        # Randomize command order to avoid fatigue bias
        import random
        command_order = self.commands.copy()
        random.shuffle(command_order)
        
        print(f"\n📝 Collection order: {command_order}")
        
        for i, command in enumerate(command_order):
            print(f"\n{'='*50}")
            print(f"Command {i+1}/{len(command_order)}: {command}")
            print(f"{'='*50}")
            
            self.collect_command_data(command, samples_per_command)
            
            # Rest between commands
            if i < len(command_order) - 1:
                print(f"\n☕ Take a 30-second break...")
                for remaining in range(30, 0, -1):
                    print(f"   Rest: {remaining}s remaining", end='\r')
                    time.sleep(1)
                print("\n")
    
    def analyze_session_quality(self):
        """Analyze the quality of collected data."""
        print(f"\n📊 Analyzing session quality...")
        
        if not self.collected_data:
            print("   ❌ No data to analyze")
            return
        
        # Basic statistics
        total_samples = len(self.collected_data)
        total_duration = sum(sample['duration'] for sample in self.collected_data)
        
        print(f"   📈 Total samples: {total_samples}")
        print(f"   ⏱️  Total duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        
        # Samples per command
        print(f"\n   📋 Samples per command:")
        for command in self.commands:
            count = sum(1 for sample in self.collected_data if sample['command'] == command)
            print(f"     {command:15s}: {count:2d} samples")
        
        # Data quality checks
        print(f"\n   🔍 Data quality checks:")
        
        data_shapes = [sample['data_shape'] for sample in self.collected_data]
        consistent_shape = len(set(tuple(shape) for shape in data_shapes)) == 1
        print(f"     Consistent data shape: {'✅' if consistent_shape else '❌'}")
        
        if consistent_shape:
            print(f"     Data shape: {data_shapes[0]}")
        
        # Signal quality analysis (for real data)
        if not self.simulation_mode and self.collected_data:
            # Analyze signal amplitude and noise
            sample_data = np.array(self.collected_data[0]['eeg_data'])
            signal_range = np.max(sample_data) - np.min(sample_data)
            signal_std = np.std(sample_data)
            
            print(f"     Signal range: {signal_range:.3f}")
            print(f"     Signal std: {signal_std:.3f}")
            print(f"     Signal quality: {'✅ Good' if 10 < signal_range < 200 else '⚠️  Check'}")
        
        # Update metadata with quality metrics
        self.session_metadata.update({
            'end_time': datetime.now().isoformat(),
            'total_duration_seconds': total_duration,
            'data_quality': {
                'total_samples': total_samples,
                'consistent_shape': consistent_shape,
                'simulation_mode': self.simulation_mode
            }
        })
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
        
        print(f"   💾 Session metadata saved to: {self.metadata_file}")
    
    def export_for_training(self):
        """Export data in format ready for model training."""
        print(f"\n📤 Exporting data for training...")
        
        if not self.collected_data:
            print("   ❌ No data to export")
            return
        
        # Create training format
        training_data = {
            'samples': [],
            'labels': [],
            'metadata': self.session_metadata
        }
        
        # Convert to training format
        for sample in self.collected_data:
            training_data['samples'].append(sample['eeg_data'])
            training_data['labels'].append(sample['command'])
        
        # Save training data
        training_file = self.output_dir / f"{self.session_name}_training_ready.json"
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"   💾 Training data saved to: {training_file}")
        print(f"   📊 Ready for use with train_model.py")
        
        return str(training_file)

def main():
    """Main data collection interface."""
    parser = argparse.ArgumentParser(description="Collect EEG training data")
    parser.add_argument('--session', default='training_session', help='Session name')
    parser.add_argument('--samples', type=int, default=5, help='Samples per command')
    parser.add_argument('--duration', type=int, default=4, help='Trial duration in seconds')
    parser.add_argument('--commands', nargs='+', help='Specific commands to collect')
    
    args = parser.parse_args()
    
    print("🗂️  EEG Training Data Collection")
    print("=" * 50)
    print(f"Session: {args.session}")
    print(f"Samples per command: {args.samples}")
    print(f"Trial duration: {args.duration}s")
    
    # Initialize collector
    collector = TrainingDataCollector(args.session)
    
    try:
        if args.commands:
            # Collect specific commands
            for command in args.commands:
                if command in collector.commands:
                    collector.collect_command_data(command, args.samples, args.duration)
                else:
                    print(f"❌ Unknown command: {command}")
        else:
            # Collect balanced dataset
            collector.collect_balanced_dataset(args.samples)
        
        # Analyze and export
        collector.analyze_session_quality()
        training_file = collector.export_for_training()
        
        print(f"\n✅ Data collection completed!")
        print(f"📁 Session file: {collector.session_file}")
        print(f"📁 Training file: {training_file}")
        print(f"📁 Metadata: {collector.metadata_file}")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review data quality in metadata file")
        print(f"   2. Run: python3 training/train_model.py --data {training_file}")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Collection interrupted by user")
        collector.analyze_session_quality()
        print(f"📁 Partial data saved to: {collector.session_file}")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
