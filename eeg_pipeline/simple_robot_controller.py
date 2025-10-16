#!/usr/bin/env python3
"""
Simple EEG → Robot Controller (No AI Model Required)

Uses basic band power analysis to control robot arm:
- High alpha (8-13 Hz): Relaxation → Move forward
- High beta (13-30 Hz): Concentration → Move up  
- High theta (4-8 Hz): Drowsiness → Move down
- Balanced: Stay in place

This demonstrates the pipeline without requiring a trained AI model.
"""

import argparse
import json
import logging
import sys
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

from kafka import KafkaConsumer, KafkaProducer

from schemas.eeg_schemas import WindowBandPower
from schemas.robot_schemas import RobotCommand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simple_robot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class RobotState:
    """Simple robot state"""
    x: float = 0.0  # Forward/back position (mm)
    y: float = 0.0  # Left/right position (mm)
    z: float = 300.0  # Height (mm)
    gripper_open: bool = True
    
    velocity_limit: float = 50.0  # mm/s
    position_limits: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.position_limits is None:
            self.position_limits = {
                'x': (-500, 500),
                'y': (-500, 500),
                'z': (0, 600)
            }
    
    def update(self, dx: float, dy: float, dz: float, dt: float = 0.1):
        """Update position with velocity limits"""
        max_delta = self.velocity_limit * dt
        
        dx = max(-max_delta, min(max_delta, dx))
        dy = max(-max_delta, min(max_delta, dy))
        dz = max(-max_delta, min(max_delta, dz))
        
        self.x = max(self.position_limits['x'][0], 
                    min(self.position_limits['x'][1], self.x + dx))
        self.y = max(self.position_limits['y'][0], 
                    min(self.position_limits['y'][1], self.y + dy))
        self.z = max(self.position_limits['z'][0], 
                    min(self.position_limits['z'][1], self.z + dz))


class SimpleBandPowerController:
    """
    Convert EEG band powers to robot commands using simple rules.
    
    Rules:
    - Dominant Alpha (8-13 Hz) → Move forward (relaxed focus)
    - Dominant Beta (13-30 Hz) → Move up (active concentration)
    - Dominant Theta (4-8 Hz) → Move down (low attention)
    - Dominant Delta (0.5-4 Hz) → Move back (deep relaxation)
    """
    
    def __init__(self, movement_scale: float = 20.0):
        self.movement_scale = movement_scale
        self.state = RobotState()
        self.command_count = 0
        
        logger.info("Simple Band Power Controller initialized")
        logger.info(f"Movement scale: {movement_scale} mm/s")
    
    def analyze_band_powers(self, band_power: WindowBandPower) -> Dict[str, float]:
        """
        Analyze band power averages across all channels.
        
        Returns:
            dict: Average power for each band
        """
        # Average across all channels for each band
        avg_powers = {}
        
        if band_power.delta:
            avg_powers['delta'] = sum(band_power.delta) / len(band_power.delta)
        if band_power.theta:
            avg_powers['theta'] = sum(band_power.theta) / len(band_power.theta)
        if band_power.alpha:
            avg_powers['alpha'] = sum(band_power.alpha) / len(band_power.alpha)
        if band_power.beta:
            avg_powers['beta'] = sum(band_power.beta) / len(band_power.beta)
        if band_power.gamma:
            avg_powers['gamma'] = sum(band_power.gamma) / len(band_power.gamma)
        
        return avg_powers
    
    def find_dominant_band(self, avg_powers: Dict[str, float]) -> str:
        """Find which frequency band has highest power"""
        if not avg_powers:
            return "none"
        
        return max(avg_powers.items(), key=lambda x: x[1])[0]
    
    def band_to_movement(self, dominant_band: str, power_ratio: float) -> tuple:
        """
        Convert dominant band to movement deltas.
        
        Args:
            dominant_band: Name of dominant frequency band
            power_ratio: Ratio of dominant to average (strength indicator)
            
        Returns:
            (dx, dy, dz): Movement deltas in mm
        """
        # Scale movement by how dominant the band is
        strength = min(power_ratio - 1.0, 1.0)  # 0 to 1 scale
        move = self.movement_scale * strength
        
        movements = {
            'delta': (-move, 0, 0),      # Move back (deep rest)
            'theta': (0, 0, -move),      # Move down (drowsy)
            'alpha': (move, 0, 0),       # Move forward (relaxed)
            'beta': (0, 0, move),        # Move up (focused)
            'gamma': (0, move, 0),       # Move right (hyper-alert)
        }
        
        return movements.get(dominant_band, (0, 0, 0))
    
    def process_band_power(self, band_power: WindowBandPower) -> RobotCommand:
        """
        Convert band power to robot command.
        
        Args:
            band_power: WindowBandPower from consumer
            
        Returns:
            RobotCommand to control robot
        """
        self.command_count += 1
        
        # Analyze band powers
        avg_powers = self.analyze_band_powers(band_power)
        
        if not avg_powers:
            logger.warning("No band powers computed")
            return self._create_hold_command()
        
        # Find dominant band
        dominant_band = self.find_dominant_band(avg_powers)
        avg_power = sum(avg_powers.values()) / len(avg_powers)
        power_ratio = avg_powers[dominant_band] / avg_power if avg_power > 0 else 1.0
        
        # Convert to movement
        dx, dy, dz = self.band_to_movement(dominant_band, power_ratio)
        
        # Update state
        self.state.update(dx, dy, dz)
        
        # Log every 10th command
        if self.command_count % 10 == 0:
            logger.info(
                f"Command #{self.command_count}: "
                f"Dominant={dominant_band} (ratio={power_ratio:.2f}) → "
                f"Position=({self.state.x:.1f}, {self.state.y:.1f}, {self.state.z:.1f})"
            )
        
        # Create robot command
        return self._create_position_command()
    
    def _create_position_command(self) -> RobotCommand:
        """Create command to move to current state position"""
        return RobotCommand(
            command_type="move_to_position",
            parameters={
                "x": self.state.x,
                "y": self.state.y,
                "z": self.state.z,
                "speed": 0.5
            },
            metadata={
                "control_mode": "band_power",
                "command_number": self.command_count
            }
        )
    
    def _create_hold_command(self) -> RobotCommand:
        """Create command to hold current position"""
        return RobotCommand(
            command_type="hold_position",
            parameters={},
            metadata={"reason": "no_band_power_data"}
        )


def main():
    parser = argparse.ArgumentParser(
        description="Simple EEG Band Power → Robot Controller"
    )
    parser.add_argument(
        '--bootstrap-servers',
        default='localhost:9092',
        help='Kafka bootstrap servers'
    )
    parser.add_argument(
        '--input-topic',
        default='band-power',
        help='Input topic with band power data'
    )
    parser.add_argument(
        '--output-topic',
        default='robot-commands',
        help='Output topic for robot commands'
    )
    parser.add_argument(
        '--movement-scale',
        type=float,
        default=20.0,
        help='Movement speed scale (mm/s)'
    )
    parser.add_argument(
        '--consumer-group',
        default='simple-robot-controller',
        help='Kafka consumer group'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Simple Band Power → Robot Controller")
    logger.info("=" * 60)
    logger.info(f"Input topic: {args.input_topic}")
    logger.info(f"Output topic: {args.output_topic}")
    logger.info(f"Movement scale: {args.movement_scale} mm/s")
    logger.info("=" * 60)
    
    # Initialize controller
    controller = SimpleBandPowerController(movement_scale=args.movement_scale)
    
    # Create Kafka consumer
    logger.info("Connecting to Kafka...")
    consumer = KafkaConsumer(
        args.input_topic,
        bootstrap_servers=args.bootstrap_servers,
        group_id=args.consumer_group,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    
    # Create Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    logger.info("✅ Connected to Kafka")
    logger.info("🎮 Waiting for band power data...")
    logger.info("")
    
    try:
        message_count = 0
        
        for message in consumer:
            try:
                message_count += 1
                
                # Parse band power
                band_power_dict = message.value
                band_power = WindowBandPower(**band_power_dict)
                
                # Process with controller
                robot_command = controller.process_band_power(band_power)
                
                # Send to robot
                producer.send(
                    args.output_topic,
                    value=robot_command.model_dump()
                )
                
                # Log progress
                if message_count % 50 == 0:
                    logger.info(f"Processed {message_count} messages")
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue
                
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
    finally:
        consumer.close()
        producer.close()
        
        logger.info("=" * 60)
        logger.info(f"Total messages processed: {message_count}")
        logger.info(f"Total commands generated: {controller.command_count}")
        logger.info(f"Final position: ({controller.state.x:.1f}, "
                   f"{controller.state.y:.1f}, {controller.state.z:.1f})")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
