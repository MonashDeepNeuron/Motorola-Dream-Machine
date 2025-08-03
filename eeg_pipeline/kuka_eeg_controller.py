#!/usr/bin/env python3
"""
KUKA EEG Controller

This script connects to the EEG pipeline and controls a KUKA robotic arm
based on real-time brain activity analysis.

Usage:
    python kuka_eeg_controller.py --kafka-server 192.168.1.100:9092 --kuka-ip 192.168.1.200
"""

import json
import time
import argparse
from typing import Dict, Any
from kafka import KafkaConsumer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockKukaArm:
    """Mock KUKA arm for testing without real hardware"""
    
    def __init__(self, ip_address: str):
        self.ip = ip_address
        self.position = "home"
        logger.info(f"Mock KUKA connected to {ip_address}")
    
    def move_to_position(self, position: str):
        self.position = position
        logger.info(f"KUKA moving to: {position}")
    
    def move_left(self):
        self.move_to_position("left")
    
    def move_right(self):
        self.move_to_position("right")
    
    def stop(self):
        self.move_to_position("stop")
    
    def get_status(self):
        return {"position": self.position, "ready": True}

# Uncomment and modify this section when you have real KUKA libraries
"""
try:
    from kuka_python_api import KukaRobot  # Replace with your actual KUKA library
    
    class RealKukaArm:
        def __init__(self, ip_address: str):
            self.robot = KukaRobot(ip_address)
            self.robot.connect()
            logger.info(f"Real KUKA connected to {ip_address}")
        
        def move_to_position(self, position: str):
            position_map = {
                "rest": [0, -90, 90, 0, 90, 0],      # Safe rest position
                "ready": [0, -45, 45, 0, 45, 0],     # Ready position
                "active": [45, -45, 45, 0, 45, 0],   # Active position
                "left": [-30, -45, 45, 0, 45, 0],    # Left gesture
                "right": [30, -45, 45, 0, 45, 0],    # Right gesture
                "stop": [0, -90, 90, 0, 90, 0]       # Stop position
            }
            
            if position in position_map:
                self.robot.move_to_joint_position(position_map[position])
                logger.info(f"KUKA moving to: {position}")
            else:
                logger.warning(f"Unknown position: {position}")
        
        def move_left(self):
            self.move_to_position("left")
        
        def move_right(self):
            self.move_to_position("right")
        
        def stop(self):
            self.move_to_position("stop")
        
        def get_status(self):
            return self.robot.get_status()

except ImportError:
    logger.info("📦 KUKA library not found - using mock arm for testing")
    RealKukaArm = MockKukaArm
"""

RealKukaArm = MockKukaArm

class EEGKukaController:
    """Main controller that processes EEG data and controls KUKA arm"""
    
    def __init__(self, kafka_server: str, kuka_ip: str, use_mock: bool = True):
        self.kafka_server = kafka_server
        self.kuka_ip = kuka_ip
        
        # Initialize KUKA arm
        if use_mock:
            self.kuka = MockKukaArm(kuka_ip)
        else:
            self.kuka = RealKukaArm(kuka_ip)
        
        # Initialize Kafka consumers
        self.setup_kafka_consumers()
        
        # Control parameters
        self.alpha_threshold_high = 0.7  # High relaxation
        self.alpha_threshold_low = 0.3   # High focus
        self.last_command_time = 0
        self.command_cooldown = 1.0      # Seconds between commands
        
        logger.info("✅ EEG-KUKA Controller initialized")
    
    def setup_kafka_consumers(self):
        """Set up Kafka consumers for different data types"""
        
        # Consumer for real-time EEG samples (motor imagery detection)
        self.eeg_consumer = KafkaConsumer(
            'raw-eeg',
            bootstrap_servers=[self.kafka_server],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            consumer_timeout_ms=1000
        )
        
        # Consumer for band power analysis (relaxation/focus detection)
        self.bandpower_consumer = KafkaConsumer(
            'eeg-bandpower',
            bootstrap_servers=[self.kafka_server],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',
            consumer_timeout_ms=1000
        )
        
        logger.info(f"📡 Connected to Kafka at {self.kafka_server}")
    
    def process_motor_imagery(self, eeg_sample: Dict[str, Any]):
        """Process motor imagery events for direct control"""
        
        if 'event_annotation' in eeg_sample:
            event = eeg_sample['event_annotation']
            
            # Check cooldown
            current_time = time.time()
            if current_time - self.last_command_time < self.command_cooldown:
                return
            
            if event == 'T1':  # Left hand imagery
                logger.info("Motor imagery: LEFT HAND detected")
                self.kuka.move_left()
                self.last_command_time = current_time
                
            elif event == 'T2':  # Right hand imagery
                logger.info("Motor imagery: RIGHT HAND detected")
                self.kuka.move_right()
                self.last_command_time = current_time
                
            elif event == 'T0':  # Rest
                logger.info("Rest state detected")
                self.kuka.stop()
                self.last_command_time = current_time
    
    def process_band_power(self, band_power: Dict[str, Any]):
        """Process frequency band power for relaxation/focus control"""
        
        # Extract alpha power (relaxation indicator)
        alpha_power = band_power.get('Alpha', 0)
        beta_power = band_power.get('Beta', 0)
        
        # Calculate focus/relaxation state
        focus_ratio = beta_power / (alpha_power + 0.001)  # Avoid division by zero
        
        logger.info(f"Alpha: {alpha_power:.3f}, Beta: {beta_power:.3f}, Focus: {focus_ratio:.3f}")
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return
        
        # Control logic based on mental state
        if alpha_power > self.alpha_threshold_high:
            logger.info("High relaxation detected - moving to rest")
            self.kuka.move_to_position("rest")
            self.last_command_time = current_time
            
        elif alpha_power < self.alpha_threshold_low:
            # High focus - move to active position
            logger.info(" High focus detected - moving to active")
            self.kuka.move_to_position("active")
            self.last_command_time = current_time
            
        else:
            # Medium state - ready position
            if time.time() - self.last_command_time > 5.0:  # Only update every 5 seconds for medium
                logger.info(" Medium state - ready position")
                self.kuka.move_to_position("ready")
                self.last_command_time = current_time
    
    def run_motor_imagery_mode(self):
        """Run in motor imagery control mode"""
        logger.info("🎮 Starting Motor Imagery Control Mode")
        logger.info("   T1 = Left hand imagery → Move left")
        logger.info("   T2 = Right hand imagery → Move right")
        logger.info("   T0 = Rest → Stop")
        
        try:
            for message in self.eeg_consumer:
                eeg_sample = message.value
                self.process_motor_imagery(eeg_sample)
                
        except KeyboardInterrupt:
            logger.info("Motor imagery mode stopped by user")
    
    def run_relaxation_mode(self):
        """Run in relaxation/focus control mode"""
        logger.info("🧘 Starting Relaxation/Focus Control Mode")
        logger.info(f"   Alpha > {self.alpha_threshold_high} → Rest position")
        logger.info(f"   Alpha < {self.alpha_threshold_low} → Active position")
        logger.info(f"   Alpha medium → Ready position")
        
        try:
            for message in self.bandpower_consumer:
                band_power = message.value
                self.process_band_power(band_power)
                
        except KeyboardInterrupt:
            logger.info("Relaxation mode stopped by user")
    
    def run_combined_mode(self):
        """Run both motor imagery and relaxation control"""
        logger.info(" Starting Combined Control Mode")
        logger.info("   Processing both motor imagery and relaxation signals")
        
        try:
            while True:
                # Check for motor imagery events
                try:
                    eeg_message = next(self.eeg_consumer)
                    self.process_motor_imagery(eeg_message.value)
                except StopIteration:
                    pass
                
                # Check for band power updates
                try:
                    bp_message = next(self.bandpower_consumer)
                    self.process_band_power(bp_message.value)
                except StopIteration:
                    pass
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
        except KeyboardInterrupt:
            logger.info("Combined mode stopped by user")

def main():
    parser = argparse.ArgumentParser(description='EEG-controlled KUKA arm')
    parser.add_argument('--kafka-server', default='localhost:9092',
                        help='Kafka server address (default: localhost:9092)')
    parser.add_argument('--kuka-ip', default='192.168.1.200',
                        help='KUKA arm IP address (default: 192.168.1.200)')
    parser.add_argument('--mode', choices=['motor', 'relax', 'combined'], default='combined',
                        help='Control mode (default: combined)')
    parser.add_argument('--use-real-kuka', action='store_true',
                        help='Use real KUKA arm instead of mock')
    
    args = parser.parse_args()
    
    # Create controller
    controller = EEGKukaController(
        kafka_server=args.kafka_server,
        kuka_ip=args.kuka_ip,
        use_mock=not args.use_real_kuka
    )
    
    # Run selected mode
    if args.mode == 'motor':
        controller.run_motor_imagery_mode()
    elif args.mode == 'relax':
        controller.run_relaxation_mode()
    else:
        controller.run_combined_mode()

if __name__ == '__main__':
    main()
