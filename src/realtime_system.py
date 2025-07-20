#!/usr/bin/env python3
"""
Real-time EEG-to-Robot Control System
=====================================

This is the main module that integrates all components for real-time EEG-based robot control.
It connects Emotiv EEG streaming, signal processing, feature extraction, ML inference, 
and robot control into a unified system.
"""

import time
import threading
import queue
import signal
import sys
from typing import Dict, List, Optional, Any
import yaml
import logging
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json

# Import system components
from src.eeg.emotiv_streamer import EmotivStreamer
from src.eeg.processor import RealTimeEEGProcessor
from src.eeg.features import EEGFeatureExtractor
from src.model.inference import EEGModelInference
from src.robot.controller import RobotController

@dataclass
class SystemStatus:
    """System status information"""
    eeg_connected: bool
    eeg_streaming: bool
    processing_active: bool
    model_loaded: bool
    robot_active: bool
    total_processed: int
    processing_rate: float
    last_prediction: str
    last_confidence: float
    uptime: float

class RealTimeEEGRobotSystem:
    """Unified real-time EEG-to-robot control system"""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize the complete system"""
        self.config_dir = config_dir
        
        # Load configurations
        self.pipeline_config = self._load_config("pipeline.yaml")
        self.emotiv_config = self._load_config("emotiv.yaml")
        self.robot_config = self._load_config("robot.yaml")
        
        # Initialize logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.eeg_streamer = None
        self.eeg_processor = None
        self.feature_extractor = None
        self.model_inference = None
        self.robot_controller = None
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Processing pipeline
        self.data_queue = queue.Queue(maxsize=1000)
        self.processing_thread = None
        
        # Statistics
        self.total_windows_processed = 0
        self.total_predictions_made = 0
        self.total_commands_sent = 0
        self.last_prediction = "stop"
        self.last_confidence = 0.0
        
        # Performance monitoring
        self.processing_times = []
        self.max_processing_time_history = 100
        
        self.logger.info("Real-time EEG-Robot system initialized")
    
    def _load_config(self, filename: str) -> Dict[str, Any]:
        """Load configuration file"""
        config_path = f"{self.config_dir}/{filename}"
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config {config_path}: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.INFO
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Configure logging
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # File handler
        file_handler = logging.FileHandler(f"logs/eeg_robot_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def initialize_components(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Initialize EEG streamer
            self.logger.info("Initializing EEG streamer...")
            self.eeg_streamer = EmotivStreamer(f"{self.config_dir}/emotiv.yaml")
            
            # Initialize EEG processor
            self.logger.info("Initializing EEG processor...")
            self.eeg_processor = RealTimeEEGProcessor(f"{self.config_dir}/pipeline.yaml")
            
            # Initialize feature extractor
            self.logger.info("Initializing feature extractor...")
            self.feature_extractor = EEGFeatureExtractor(f"{self.config_dir}/pipeline.yaml")
            
            # Initialize model inference
            self.logger.info("Initializing model inference...")
            self.model_inference = EEGModelInference(f"{self.config_dir}/pipeline.yaml")
            
            # Load model
            if not self.model_inference.load_model():
                self.logger.warning("Model loading failed - continuing with mock inference")
            
            # Initialize robot controller
            self.logger.info("Initializing robot controller...")
            self.robot_controller = RobotController(f"{self.config_dir}/robot.yaml")
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            return False
    
    def start_system(self) -> bool:
        """Start the complete system"""
        if self.is_running:
            self.logger.warning("System already running")
            return True
        
        try:
            self.logger.info("Starting real-time EEG-Robot system...")
            
            # Initialize components if not done
            if self.eeg_streamer is None:
                if not self.initialize_components():
                    return False
            
            # Connect to EEG headset
            self.logger.info("Connecting to EEG headset...")
            if not self.eeg_streamer.connect():
                self.logger.error("Failed to connect to EEG headset")
                return False
            
            # Setup EEG processor with channel information
            channel_names = self.eeg_streamer.get_channel_names()
            self.eeg_processor.set_channels(channel_names)
            
            # Start robot controller
            self.logger.info("Starting robot controller...")
            self.robot_controller.start()
            
            # Start processing thread
            self.is_running = True
            self.start_time = time.time()
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            # Start EEG streaming with callback
            self.logger.info("Starting EEG streaming...")
            if not self.eeg_streamer.start_streaming(self._eeg_data_callback):
                self.logger.error("Failed to start EEG streaming")
                return False
            
            self.logger.info("Real-time EEG-Robot system started successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            self.stop_system()
            return False
    
    def stop_system(self):
        """Stop the complete system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping real-time EEG-Robot system...")
        
        # Stop system
        self.is_running = False
        
        # Stop EEG streaming
        if self.eeg_streamer:
            self.eeg_streamer.stop_streaming()
            self.eeg_streamer.disconnect()
        
        # Stop robot controller
        if self.robot_controller:
            self.robot_controller.stop()
        
        # Wait for processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.logger.info("System stopped")
    
    def _eeg_data_callback(self, eeg_data: np.ndarray, channels: List[str], timestamp: float):
        """Callback for new EEG data from streamer"""
        try:
            # Add sample to processor
            self.eeg_processor.add_sample(eeg_data, timestamp)
            
            # Check if we can process a window
            if self.eeg_processor.can_process_window():
                # Process window
                window_result = self.eeg_processor.process_window()
                
                if window_result:
                    # Add to processing queue
                    try:
                        self.data_queue.put(window_result, timeout=0.1)
                    except queue.Full:
                        self.logger.warning("Processing queue full, dropping window")
        
        except Exception as e:
            self.logger.error(f"EEG data callback error: {e}")
    
    def _processing_loop(self):
        """Main processing loop for feature extraction and prediction"""
        self.logger.info("Processing loop started")
        
        while self.is_running:
            try:
                # Get processed window from queue
                try:
                    window_result = self.data_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                
                # Extract features from processed window
                features_dict = self.feature_extractor.extract_features(
                    window_result['filtered_data'],
                    window_result['channel_names']
                )
                
                # Convert to feature vector
                feature_vector = self.feature_extractor.get_feature_vector(features_dict)
                
                # Make prediction
                prediction, confidence, details = self.model_inference.predict(feature_vector)
                
                # Update statistics
                self.total_windows_processed += 1
                self.total_predictions_made += 1
                self.last_prediction = details.get('class_name', 'unknown')
                self.last_confidence = confidence
                
                # Send command to robot if confidence is high enough
                if details.get('meets_threshold', False):
                    success = self.robot_controller.process_eeg_prediction(prediction, confidence)
                    if success:
                        self.total_commands_sent += 1
                        self.logger.debug(f"Robot command sent: {self.last_prediction} (confidence: {confidence:.3f})")
                
                # Record processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                if len(self.processing_times) > self.max_processing_time_history:
                    self.processing_times.pop(0)
                
                # Log periodically
                if self.total_windows_processed % 100 == 0:
                    self.logger.info(f"Processed {self.total_windows_processed} windows, "
                                   f"avg processing time: {np.mean(self.processing_times):.3f}s")
                
                # Mark task as done
                self.data_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Processing loop error: {e}")
                time.sleep(0.1)
        
        self.logger.info("Processing loop stopped")
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        uptime = time.time() - self.start_time if self.start_time else 0
        processing_rate = self.total_windows_processed / max(uptime, 1)
        
        return SystemStatus(
            eeg_connected=self.eeg_streamer.is_headset_connected() if self.eeg_streamer else False,
            eeg_streaming=self.eeg_streamer.is_streaming if self.eeg_streamer else False,
            processing_active=self.is_running,
            model_loaded=self.model_inference.is_loaded if self.model_inference else False,
            robot_active=self.robot_controller.is_running if self.robot_controller else False,
            total_processed=self.total_windows_processed,
            processing_rate=processing_rate,
            last_prediction=self.last_prediction,
            last_confidence=self.last_confidence,
            uptime=uptime
        )
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed status of all components"""
        status = {
            'system': {
                'running': self.is_running,
                'uptime': time.time() - self.start_time if self.start_time else 0,
                'windows_processed': self.total_windows_processed,
                'predictions_made': self.total_predictions_made,
                'commands_sent': self.total_commands_sent
            }
        }
        
        if self.eeg_streamer:
            status['eeg_streamer'] = self.eeg_streamer.get_connection_status()
        
        if self.eeg_processor:
            status['eeg_processor'] = self.eeg_processor.get_status()
        
        if self.model_inference:
            status['model_inference'] = self.model_inference.get_statistics()
        
        if self.robot_controller:
            status['robot_controller'] = self.robot_controller.get_status()
        
        if self.processing_times:
            status['performance'] = {
                'avg_processing_time': np.mean(self.processing_times),
                'max_processing_time': np.max(self.processing_times),
                'min_processing_time': np.min(self.processing_times)
            }
        
        return status
    
    def save_session_log(self, filename: str = None):
        """Save session information to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"logs/session_log_{timestamp}.json"
        
        try:
            session_data = {
                'session_info': {
                    'start_time': self.start_time,
                    'end_time': time.time(),
                    'duration': time.time() - self.start_time if self.start_time else 0
                },
                'statistics': {
                    'windows_processed': self.total_windows_processed,
                    'predictions_made': self.total_predictions_made,
                    'commands_sent': self.total_commands_sent
                },
                'final_status': self.get_detailed_status()
            }
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"Session log saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session log: {e}")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    print(f"\nReceived signal {signum}, shutting down...")
    if 'system' in globals():
        system.stop_system()
    sys.exit(0)

def main():
    """Main function to run the real-time system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time EEG-to-Robot Control System")
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    parser.add_argument('--duration', type=int, help='Run duration in seconds (default: run indefinitely)')
    parser.add_argument('--status-interval', type=int, default=30, help='Status report interval (seconds)')
    parser.add_argument('--save-log', action='store_true', help='Save session log on exit')
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create system
    global system
    system = RealTimeEEGRobotSystem(args.config_dir)
    
    try:
        # Start system
        print("Starting real-time EEG-to-Robot control system...")
        if not system.start_system():
            print("Failed to start system")
            return 1
        
        print("System started successfully!")
        print(f"System will run for {args.duration} seconds" if args.duration else "System running indefinitely (Ctrl+C to stop)")
        
        # Run for specified duration or indefinitely
        start_time = time.time()
        last_status_time = start_time
        
        while True:
            current_time = time.time()
            
            # Check if duration exceeded
            if args.duration and (current_time - start_time) >= args.duration:
                print(f"Duration {args.duration} seconds reached, stopping...")
                break
            
            # Print status periodically
            if current_time - last_status_time >= args.status_interval:
                status = system.get_system_status()
                print(f"\n--- System Status (Uptime: {status.uptime:.1f}s) ---")
                print(f"EEG Connected: {status.eeg_connected}, Streaming: {status.eeg_streaming}")
                print(f"Processing Active: {status.processing_active}, Model Loaded: {status.model_loaded}")
                print(f"Robot Active: {status.robot_active}")
                print(f"Windows Processed: {status.total_processed} (Rate: {status.processing_rate:.2f}/s)")
                print(f"Last Prediction: {status.last_prediction} (Confidence: {status.last_confidence:.3f})")
                
                last_status_time = current_time
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"System error: {e}")
        return 1
    
    finally:
        # Stop system
        print("Stopping system...")
        system.stop_system()
        
        # Save session log if requested
        if args.save_log:
            system.save_session_log()
        
        print("System stopped.")
        return 0

if __name__ == "__main__":
    exit(main())
