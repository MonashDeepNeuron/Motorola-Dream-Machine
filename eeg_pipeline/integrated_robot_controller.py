#!/usr/bin/env python3
"""
Integrated Robot Controller
Consumes AI predictions and controls robot arm with safety features.
"""

import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import time
from datetime import datetime, timezone
from typing import Dict, Optional
from kafka import KafkaConsumer
import logging

from schemas.robot_schemas import RobotCommand, RobotState, SafetyLimits, CommandType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobotArmController:
    """
    Generic robot arm controller that works with multiple robot types.
    """
    
    def __init__(self, robot_type: str = "mock", robot_ip: str = "localhost", 
                 safety_limits: Optional[SafetyLimits] = None):
        self.robot_type = robot_type
        self.robot_ip = robot_ip
        self.safety = safety_limits or SafetyLimits()
        self.last_command_time = time.time()
        self.current_state = self._get_initial_state()
        
        # Initialize robot connection
        if robot_type == "mock":
            self.robot = MockRobot(robot_ip)
        elif robot_type == "ur":
            self.robot = URRobot(robot_ip)
        elif robot_type == "kuka":
            self.robot = KUKARobot(robot_ip)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        
        logger.info(f"✅ Initialized {robot_type} robot at {robot_ip}")
        logger.info(f"   Max velocity: {self.safety.max_velocity} m/s")
        logger.info(f"   Min confidence: {self.safety.min_confidence}")
        logger.info(f"   Command timeout: {self.safety.command_timeout_ms}ms")
    
    def _get_initial_state(self) -> RobotState:
        """Get initial robot state."""
        return RobotState(
            timestamp=datetime.now(timezone.utc),
            position=[0.3, 0.0, 0.3, 0.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            is_moving=False,
            in_safety_limits=True
        )
    
    def check_safety(self, command: RobotCommand) -> tuple[bool, str]:
        """
        Check if command is safe to execute.
        
        Returns:
            (is_safe, reason)
        """
        # Check confidence threshold
        if command.confidence < self.safety.min_confidence:
            return False, f"Confidence {command.confidence:.2f} < threshold {self.safety.min_confidence}"
        
        # Check timeout
        time_since_last = (time.time() - self.last_command_time) * 1000
        if time_since_last > self.safety.command_timeout_ms:
            return False, f"Command timeout: {time_since_last:.0f}ms > {self.safety.command_timeout_ms}ms"
        
        # Check velocity limits if provided
        if command.velocity:
            for i, v in enumerate(command.velocity[:3]):  # Check linear velocities
                if abs(v) > self.safety.max_velocity:
                    return False, f"Velocity component {i} exceeds limit: {abs(v):.3f} > {self.safety.max_velocity}"
        
        # Check workspace bounds if position provided
        if command.position:
            for i in range(6):
                if command.position[i] < self.safety.workspace_min[i] or \
                   command.position[i] > self.safety.workspace_max[i]:
                    return False, f"Position component {i} out of bounds"
        
        return True, "OK"
    
    def execute_command(self, command: RobotCommand) -> bool:
        """
        Execute robot command with safety checks.
        
        Returns:
            True if executed successfully
        """
        # Safety check
        is_safe, reason = self.check_safety(command)
        if not is_safe:
            logger.warning(f"❌ Command rejected: {reason}")
            return False
        
        # Update last command time
        self.last_command_time = time.time()
        
        # Map command to robot action
        try:
            if command.command_type == "REST" or command.command_type == "STOP":
                self.robot.stop()
                logger.info(f"🛑 STOP (confidence: {command.confidence:.2f})")
                
            elif command.command_type == "LEFT":
                velocity = [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Move left
                self.robot.move_velocity(velocity)
                logger.info(f"⬅️  LEFT (confidence: {command.confidence:.2f})")
                
            elif command.command_type == "RIGHT":
                velocity = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]  # Move right
                self.robot.move_velocity(velocity)
                logger.info(f"➡️  RIGHT (confidence: {command.confidence:.2f})")
                
            elif command.command_type == "FORWARD":
                velocity = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]  # Move forward
                self.robot.move_velocity(velocity)
                logger.info(f"⬆️  FORWARD (confidence: {command.confidence:.2f})")
                
            elif command.command_type == "BACKWARD":
                velocity = [0.0, -0.1, 0.0, 0.0, 0.0, 0.0]  # Move backward
                self.robot.move_velocity(velocity)
                logger.info(f"⬇️  BACKWARD (confidence: {command.confidence:.2f})")
                
            elif command.command_type == "EMERGENCY_STOP":
                self.robot.emergency_stop()
                logger.critical(f"🚨 EMERGENCY STOP")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            self.robot.stop()
            return False
    
    def update_state(self):
        """Update current robot state."""
        try:
            self.current_state = self.robot.get_state()
        except Exception as e:
            logger.error(f"Failed to update state: {e}")


class MockRobot:
    """Mock robot for testing."""
    
    def __init__(self, ip: str):
        self.ip = ip
        self.position = [0.3, 0.0, 0.3, 0.0, 0.0, 0.0]
        self.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        logger.info(f"📦 Mock robot initialized at {ip}")
    
    def stop(self):
        self.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def move_velocity(self, velocity):
        self.velocity = velocity
        # Simulate position update
        for i in range(3):
            self.position[i] += velocity[i] * 0.1
    
    def emergency_stop(self):
        self.stop()
    
    def get_state(self) -> RobotState:
        return RobotState(
            timestamp=datetime.now(timezone.utc),
            position=self.position,
            velocity=self.velocity,
            is_moving=any(abs(v) > 0.001 for v in self.velocity),
            in_safety_limits=True
        )


class URRobot:
    """Universal Robots controller."""
    
    def __init__(self, ip: str):
        try:
            import rtde_control
            import rtde_receive
            
            self.rtde_c = rtde_control.RTDEControlInterface(ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
            
            if not (self.rtde_c.isConnected() and self.rtde_r.isConnected()):
                raise ConnectionError("Failed to connect to UR robot")
            
            logger.info(f"🤖 Connected to UR robot at {ip}")
            
        except ImportError:
            raise ImportError("UR robot requires 'ur-rtde' package: pip install ur-rtde")
    
    def stop(self):
        self.rtde_c.speedStop()
    
    def move_velocity(self, velocity):
        # velocity: [vx, vy, vz, wx, wy, wz]
        self.rtde_c.speedL(velocity, 0.5, 1.0)  # acceleration=0.5, time=1.0s
    
    def emergency_stop(self):
        self.rtde_c.stopScript()
        self.rtde_c.stopJ([3.0] * 6)  # High deceleration
    
    def get_state(self) -> RobotState:
        return RobotState(
            timestamp=datetime.now(timezone.utc),
            position=self.rtde_r.getActualTCPPose(),
            velocity=self.rtde_r.getActualTCPSpeed(),
            is_moving=True,  # Could check velocity magnitude
            in_safety_limits=True
        )
    
    def __del__(self):
        if hasattr(self, 'rtde_c'):
            self.rtde_c.disconnect()
        if hasattr(self, 'rtde_r'):
            self.rtde_r.disconnect()


class KUKARobot:
    """KUKA robot controller (placeholder)."""
    
    def __init__(self, ip: str):
        logger.warning("⚠️  KUKA robot interface not fully implemented")
        logger.info(f"Using mock mode for KUKA at {ip}")
        self.mock = MockRobot(ip)
    
    def stop(self):
        self.mock.stop()
    
    def move_velocity(self, velocity):
        self.mock.move_velocity(velocity)
    
    def emergency_stop(self):
        self.mock.emergency_stop()
    
    def get_state(self) -> RobotState:
        return self.mock.get_state()


def main(
    kafka_servers: str,
    input_topic: str,
    robot_type: str,
    robot_ip: str,
    min_confidence: float
):
    """
    Main control loop: consume AI predictions and control robot.
    """
    print("=" * 60)
    print("🤖 Integrated Robot Controller")
    print("=" * 60)
    
    # Setup safety limits
    safety = SafetyLimits(
        max_velocity=0.2,
        max_acceleration=0.5,
        min_confidence=min_confidence,
        command_timeout_ms=2000
    )
    
    # Initialize robot controller
    controller = RobotArmController(
        robot_type=robot_type,
        robot_ip=robot_ip,
        safety_limits=safety
    )
    
    # Setup Kafka consumer
    print(f"\n📡 Connecting to Kafka: {kafka_servers}")
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=kafka_servers,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    print(f"📥 Consuming from topic: {input_topic}")
    print(f"🎯 Robot type: {robot_type}")
    print(f"🔒 Safety: min_confidence={min_confidence}, max_vel={safety.max_velocity}m/s")
    print(f"\n▶️  Control loop started. Press Ctrl+C to stop.\n")
    
    commands_executed = 0
    commands_rejected = 0
    
    try:
        for message in consumer:
            # Parse command
            try:
                # Handle both RobotCommand format and AI prediction format
                data = message.value
                
                if 'command' in data:
                    # AI prediction format
                    command = RobotCommand(
                        command_type=data['command'],
                        confidence=data.get('confidence', 0.0),
                        timestamp=datetime.fromtimestamp(data.get('timestamp', time.time()), tz=timezone.utc),
                        prediction_probabilities=data.get('probabilities')
                    )
                else:
                    # Direct RobotCommand format
                    command = RobotCommand(**data)
                
                # Execute command
                if controller.execute_command(command):
                    commands_executed += 1
                else:
                    commands_rejected += 1
                
                # Update state periodically
                if commands_executed % 10 == 0:
                    controller.update_state()
                
            except Exception as e:
                logger.error(f"Failed to process message: {e}")
                continue
    
    except KeyboardInterrupt:
        print(f"\n\n🛑 Shutdown initiated")
    finally:
        # Stop robot
        logger.info("Stopping robot...")
        controller.robot.stop()
        
        # Statistics
        total = commands_executed + commands_rejected
        print(f"\n📊 Statistics:")
        print(f"   Commands executed: {commands_executed}/{total} ({100*commands_executed/max(1,total):.1f}%)")
        print(f"   Commands rejected: {commands_rejected}/{total}")
        
        consumer.close()
        print("✅ Controller stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated robot controller for EEG-based control"
    )
    parser.add_argument(
        "--kafka-servers", default="localhost:9092",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--input-topic", default="robot-commands",
        help="Kafka topic to consume robot commands from"
    )
    parser.add_argument(
        "--robot-type", choices=["mock", "ur", "kuka"], default="mock",
        help="Type of robot arm"
    )
    parser.add_argument(
        "--robot-ip", default="192.168.1.200",
        help="IP address of robot"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum confidence to execute commands (0-1)"
    )
    
    args = parser.parse_args()
    
    main(
        kafka_servers=args.kafka_servers,
        input_topic=args.input_topic,
        robot_type=args.robot_type,
        robot_ip=args.robot_ip,
        min_confidence=args.min_confidence
    )
