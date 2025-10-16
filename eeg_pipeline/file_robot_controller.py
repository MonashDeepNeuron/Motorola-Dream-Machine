#!/usr/bin/env python3
"""
File-Based Robot Controller

Reads robot commands from a JSON file (one command per line)
and executes them on the robot arm (mock/UR/KUKA).

Usage:
    # Terminal 1: Generate commands
    python realtime_eeg_to_robot.py --edf-file data.edf --output commands.json
    
    # Terminal 2: Execute commands
    python file_robot_controller.py --file commands.json --robot-type mock
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

from schemas.robot_schemas import RobotCommand
from integrated_robot_controller import MockRobot, URRobot, KUKARobot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FileRobotController:
    """
    Reads robot commands from file and executes them.
    
    Follows the file in real-time (like tail -f).
    """
    
    def __init__(self, robot_type: str = 'mock', robot_ip: str = '192.168.1.200'):
        logger.info(f"Initializing {robot_type.upper()} robot...")
        
        if robot_type == 'mock':
            self.robot = MockRobot()
        elif robot_type == 'ur':
            self.robot = URRobot(robot_ip)
        elif robot_type == 'kuka':
            self.robot = KUKARobot(robot_ip)
        else:
            raise ValueError(f"Unknown robot type: {robot_type}")
        
        self.robot.connect()
        self.commands_executed = 0
        
        logger.info(f"✅ {robot_type.upper()} robot connected")
    
    def execute_command(self, command_dict: dict) -> bool:
        """
        Execute a robot command.
        
        Args:
            command_dict: Command dictionary
            
        Returns:
            True if executed successfully
        """
        try:
            # Parse command
            command = RobotCommand(**command_dict)
            
            # Execute based on type
            if command.command_type == "move_to_position":
                params = command.parameters
                success = self.robot.move_to_position(
                    x=params.get('x', 0),
                    y=params.get('y', 0),
                    z=params.get('z', 300),
                    speed=params.get('speed', 0.5)
                )
            elif command.command_type == "move_joints":
                success = self.robot.move_joints(
                    command.parameters.get('joint_angles', [])
                )
            elif command.command_type == "set_gripper":
                success = self.robot.set_gripper(
                    command.parameters.get('open', True)
                )
            elif command.command_type == "hold_position":
                logger.info("Holding position")
                success = True
            elif command.command_type == "emergency_stop":
                logger.warning("⚠️  Emergency stop!")
                self.robot.emergency_stop()
                success = True
            else:
                logger.warning(f"Unknown command type: {command.command_type}")
                success = False
            
            if success:
                self.commands_executed += 1
                
                # Log every 10th command
                if self.commands_executed % 10 == 0:
                    logger.info(
                        f"Command #{self.commands_executed}: {command.command_type} "
                        f"(mode: {command.metadata.get('control_mode', 'unknown')})"
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return False
    
    def follow_file(self, file_path: str, poll_interval: float = 0.1):
        """
        Follow a file in real-time and execute commands as they arrive.
        
        Args:
            file_path: Path to JSON commands file
            poll_interval: How often to check for new commands (seconds)
        """
        path = Path(file_path)
        
        logger.info("=" * 60)
        logger.info("File-Based Robot Controller")
        logger.info("=" * 60)
        logger.info(f"Reading from: {file_path}")
        logger.info(f"Poll interval: {poll_interval}s")
        logger.info("=" * 60)
        logger.info("")
        logger.info("🤖 Waiting for commands... (Ctrl+C to stop)")
        logger.info("")
        
        # Wait for file to exist
        while not path.exists():
            logger.info(f"Waiting for {file_path} to be created...")
            time.sleep(1)
        
        # Open file and seek to end (or start)
        with open(file_path, 'r') as f:
            # Start from beginning
            f.seek(0)
            
            line_count = 0
            
            while True:
                line = f.readline()
                
                if line:
                    line_count += 1
                    line = line.strip()
                    
                    if line:
                        try:
                            command = json.loads(line)
                            self.execute_command(command)
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON on line {line_count}: {e}")
                else:
                    # No new data, wait a bit
                    time.sleep(poll_interval)
    
    def disconnect(self):
        """Disconnect from robot"""
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Commands executed: {self.commands_executed}")
        logger.info("Disconnecting from robot...")
        self.robot.disconnect()
        logger.info("✅ Disconnected")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="File-based robot controller"
    )
    parser.add_argument(
        '--file',
        required=True,
        help='JSON file with robot commands (one per line)'
    )
    parser.add_argument(
        '--robot-type',
        choices=['mock', 'ur', 'kuka'],
        default='mock',
        help='Type of robot arm'
    )
    parser.add_argument(
        '--robot-ip',
        default='192.168.1.200',
        help='IP address of robot (for UR/KUKA)'
    )
    parser.add_argument(
        '--poll-interval',
        type=float,
        default=0.1,
        help='File poll interval in seconds'
    )
    
    args = parser.parse_args()
    
    try:
        controller = FileRobotController(
            robot_type=args.robot_type,
            robot_ip=args.robot_ip
        )
        
        controller.follow_file(
            file_path=args.file,
            poll_interval=args.poll_interval
        )
        
    except KeyboardInterrupt:
        logger.info("\n⏸️  Stopped by user")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    finally:
        if 'controller' in locals():
            controller.disconnect()
    
    return 0


if __name__ == '__main__':
    exit(main())
