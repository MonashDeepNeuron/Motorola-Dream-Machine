#!/usr/bin/env python3
"""
Emotiv Real-time EEG Streaming Interface
========================================

This module provides real-time EEG data streaming from Emotiv headsets
using the Emotiv SDK. Supports EPOC X, FLEX, and Insight devices.
"""

import time
import numpy as np
import threading
import queue
from typing import Optional, Dict, List, Callable, Any
from collections import deque
import yaml
import json
import logging

try:
    from cortex import Cortex
    CORTEX_AVAILABLE = True
except ImportError:
    CORTEX_AVAILABLE = False
    print("Warning: Emotiv Cortex SDK not installed. Install from: https://www.emotiv.com/developer/")

class EmotivStreamer:
    """Real-time EEG streaming from Emotiv headsets"""
    
    def __init__(self, config_path: str = "config/emotiv.yaml"):
        """Initialize Emotiv streamer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Streaming state
        self.is_streaming = False
        self.is_connected = False
        
        # Data buffers
        self.eeg_buffer = deque(maxlen=10000)  # Store last 10000 samples
        self.quality_buffer = deque(maxlen=1000)
        
        # Threading
        self.stream_thread = None
        self.data_queue = queue.Queue(maxsize=1000)
        
        # Callbacks
        self.data_callback = None
        self.quality_callback = None
        
        # Session info
        self.session_id = None
        self.headset_id = None
        
        # Initialize Cortex if available
        if CORTEX_AVAILABLE:
            self.cortex = None
            self._initialize_cortex()
        else:
            self.cortex = None
            self.logger.warning("Cortex SDK not available - using simulation mode")
    
    def _initialize_cortex(self):
        """Initialize Cortex connection"""
        try:
            auth_config = self.config['authentication']
            self.cortex = Cortex(
                client_id=auth_config['client_id'],
                client_secret=auth_config['client_secret'],
                debug_mode=self.config.get('advanced', {}).get('debug', {}).get('verbose_logging', False)
            )
            self.logger.info("Cortex initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Cortex: {e}")
            self.cortex = None
    
    def connect(self) -> bool:
        """Connect to Emotiv headset"""
        if not CORTEX_AVAILABLE or not self.cortex:
            self.logger.warning("Using simulation mode - no real headset connection")
            self.is_connected = True
            return True
        
        try:
            # Authenticate
            auth_config = self.config['authentication']
            self.cortex.do_prepare_steps(
                username=auth_config['username'],
                password=auth_config['password']
            )
            
            # Find and connect to headset
            headsets = self.cortex.query_headsets()
            if not headsets:
                self.logger.error("No headsets found")
                return False
            
            # Use specified headset or first available
            headset_config = self.config['headset']
            if headset_config.get('device_id'):
                target_headset = next(
                    (h for h in headsets if h['id'] == headset_config['device_id']), 
                    None
                )
            else:
                target_headset = headsets[0]
            
            if not target_headset:
                self.logger.error("Target headset not found")
                return False
            
            self.headset_id = target_headset['id']
            self.cortex.connect_headset(self.headset_id)
            
            # Create session
            self.session_id = self.cortex.create_session(
                activate=True,
                headset_id=self.headset_id
            )
            
            self.is_connected = True
            self.logger.info(f"Connected to headset: {target_headset['id']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from headset"""
        self.stop_streaming()
        
        if self.cortex and self.session_id:
            try:
                self.cortex.close_session()
                self.cortex.disconnect_headset()
                self.logger.info("Disconnected from headset")
            except Exception as e:
                self.logger.error(f"Disconnection error: {e}")
        
        self.is_connected = False
        self.session_id = None
        self.headset_id = None
    
    def start_streaming(self, data_callback: Optional[Callable] = None):
        """Start EEG data streaming"""
        if not self.is_connected:
            self.logger.error("Not connected to headset")
            return False
        
        if self.is_streaming:
            self.logger.warning("Already streaming")
            return True
        
        self.data_callback = data_callback
        self.is_streaming = True
        
        if CORTEX_AVAILABLE and self.cortex:
            # Start real streaming
            self.stream_thread = threading.Thread(target=self._real_streaming_loop)
        else:
            # Start simulation streaming
            self.stream_thread = threading.Thread(target=self._simulation_streaming_loop)
        
        self.stream_thread.daemon = True
        self.stream_thread.start()
        
        self.logger.info("Started EEG streaming")
        return True
    
    def stop_streaming(self):
        """Stop EEG data streaming"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=5)
        
        if CORTEX_AVAILABLE and self.cortex:
            try:
                self.cortex.unsubscribe(['eeg'])
            except Exception as e:
                self.logger.error(f"Error stopping stream: {e}")
        
        self.logger.info("Stopped EEG streaming")
    
    def _real_streaming_loop(self):
        """Real streaming loop using Cortex SDK"""
        try:
            # Subscribe to EEG stream
            stream_config = self.config['streaming']
            streams = []
            if stream_config['streams']['eeg']:
                streams.append('eeg')
            if stream_config['streams']['motion']:
                streams.append('mot')
            
            self.cortex.subscribe(streams)
            
            # Get channel names
            channels = self.config['headset']['channels']['eeg_channels']
            
            while self.is_streaming:
                try:
                    # Get data from Cortex
                    data = self.cortex.get_data()
                    
                    if data and 'eeg' in data:
                        # Process EEG data
                        eeg_data = data['eeg']
                        timestamp = time.time()
                        
                        # Convert to numpy array
                        eeg_array = np.array(eeg_data[2:])  # Skip sequence number and timestamp
                        
                        # Store in buffer
                        self.eeg_buffer.append({
                            'timestamp': timestamp,
                            'data': eeg_array,
                            'channels': channels
                        })
                        
                        # Call callback if provided
                        if self.data_callback:
                            self.data_callback(eeg_array, channels, timestamp)
                        
                        # Check contact quality
                        self._check_contact_quality()
                    
                    time.sleep(0.001)  # Small delay to prevent CPU overload
                    
                except Exception as e:
                    self.logger.error(f"Streaming error: {e}")
                    if not self.is_streaming:
                        break
                    time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Streaming loop error: {e}")
        finally:
            self.is_streaming = False
    
    def _simulation_streaming_loop(self):
        """Simulation streaming loop for testing without hardware"""
        channels = self.config['headset']['channels']['eeg_channels']
        n_channels = len(channels)
        sample_rate = self.config['streaming']['eeg_rate']
        
        self.logger.info("Starting EEG simulation")
        
        while self.is_streaming:
            try:
                # Generate realistic EEG-like data
                timestamp = time.time()
                
                # Create synthetic EEG signal
                eeg_data = self._generate_synthetic_eeg(n_channels)
                
                # Store in buffer
                self.eeg_buffer.append({
                    'timestamp': timestamp,
                    'data': eeg_data,
                    'channels': channels
                })
                
                # Call callback if provided
                if self.data_callback:
                    self.data_callback(eeg_data, channels, timestamp)
                
                # Simulate sampling rate
                time.sleep(1.0 / sample_rate)
                
            except Exception as e:
                self.logger.error(f"Simulation error: {e}")
                break
        
        self.logger.info("EEG simulation stopped")
    
    def _generate_synthetic_eeg(self, n_channels: int) -> np.ndarray:
        """Generate realistic synthetic EEG data"""
        # Mix of different frequency components
        t = time.time()
        
        # Base frequencies
        alpha_freq = 10  # Hz
        beta_freq = 20   # Hz
        theta_freq = 6   # Hz
        
        # Generate signal for each channel
        eeg_data = np.zeros(n_channels)
        
        for ch in range(n_channels):
            # Different phase for each channel
            phase_offset = ch * 0.1
            
            # Mix of frequency components
            alpha = 20 * np.sin(2 * np.pi * alpha_freq * t + phase_offset)
            beta = 10 * np.sin(2 * np.pi * beta_freq * t + phase_offset)
            theta = 15 * np.sin(2 * np.pi * theta_freq * t + phase_offset)
            
            # Add noise
            noise = np.random.normal(0, 5)
            
            # Combine (values in microvolts)
            eeg_data[ch] = alpha + beta + theta + noise
        
        return eeg_data
    
    def _check_contact_quality(self):
        """Check electrode contact quality"""
        if not CORTEX_AVAILABLE or not self.cortex:
            return
        
        try:
            # Get contact quality
            quality = self.cortex.get_contact_quality()
            
            if quality:
                # Store quality data
                self.quality_buffer.append({
                    'timestamp': time.time(),
                    'quality': quality
                })
                
                # Check if quality callback is provided
                if self.quality_callback:
                    self.quality_callback(quality)
                
                # Check quality thresholds
                quality_config = self.config['quality']
                min_quality = quality_config['contact_quality']['threshold']
                
                poor_contacts = [
                    ch for ch, q in quality.items() 
                    if isinstance(q, (int, float)) and q < min_quality
                ]
                
                if poor_contacts:
                    self.logger.warning(f"Poor contact quality: {poor_contacts}")
        
        except Exception as e:
            self.logger.error(f"Quality check error: {e}")
    
    def get_latest_data(self, n_samples: int = 1) -> Optional[np.ndarray]:
        """Get latest EEG data samples"""
        if len(self.eeg_buffer) < n_samples:
            return None
        
        # Get last n_samples from buffer
        samples = list(self.eeg_buffer)[-n_samples:]
        
        # Extract data arrays
        data_arrays = [sample['data'] for sample in samples]
        
        # Stack into array: (n_samples, n_channels)
        return np.array(data_arrays)
    
    def get_channel_names(self) -> List[str]:
        """Get EEG channel names"""
        return self.config['headset']['channels']['eeg_channels']
    
    def get_sample_rate(self) -> int:
        """Get EEG sampling rate"""
        return self.config['streaming']['eeg_rate']
    
    def is_headset_connected(self) -> bool:
        """Check if headset is connected"""
        return self.is_connected
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get detailed connection status"""
        return {
            'connected': self.is_connected,
            'streaming': self.is_streaming,
            'headset_id': self.headset_id,
            'session_id': self.session_id,
            'buffer_size': len(self.eeg_buffer),
            'sample_rate': self.get_sample_rate(),
            'channels': self.get_channel_names()
        }

def main():
    """Test the Emotiv streamer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Emotiv EEG Streamer")
    parser.add_argument('--config', default='config/emotiv.yaml', help='Config file path')
    parser.add_argument('--duration', type=int, default=30, help='Test duration (seconds)')
    parser.add_argument('--test-connection', action='store_true', help='Test connection only')
    
    args = parser.parse_args()
    
    # Initialize streamer
    streamer = EmotivStreamer(args.config)
    
    def data_callback(eeg_data, channels, timestamp):
        print(f"Received EEG data: {len(channels)} channels, range: {eeg_data.min():.2f} to {eeg_data.max():.2f} μV")
    
    def quality_callback(quality):
        print(f"Contact quality: {quality}")
    
    try:
        # Connect to headset
        print("Connecting to Emotiv headset...")
        if not streamer.connect():
            print("Failed to connect to headset")
            return
        
        print(f"Connection status: {streamer.get_connection_status()}")
        
        if args.test_connection:
            print("Connection test successful!")
            return
        
        # Start streaming
        print(f"Starting EEG streaming for {args.duration} seconds...")
        streamer.data_callback = data_callback
        streamer.quality_callback = quality_callback
        
        if not streamer.start_streaming():
            print("Failed to start streaming")
            return
        
        # Stream for specified duration
        time.sleep(args.duration)
        
        # Show statistics
        print(f"\nStreaming completed!")
        print(f"Buffer size: {len(streamer.eeg_buffer)} samples")
        
        # Get latest data
        latest = streamer.get_latest_data(10)
        if latest is not None:
            print(f"Latest data shape: {latest.shape}")
            print(f"Data range: {latest.min():.2f} to {latest.max():.2f} μV")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("Disconnecting...")
        streamer.disconnect()
        print("Done!")

if __name__ == "__main__":
    main()
