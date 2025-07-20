#!/usr/bin/env python3
"""
Robot Control Module
====================

This module provides robot control capabilities for the Motorola Dream Machine,
supporting UR robot simulation and real robot control through EEG-based commands.
"""

import json
import numpy as np
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import logging
from dataclasses import dataclass
from enum import Enum
import math

class RobotCommand(Enum):
    """Available robot commands"""
    STOP = "stop"
    MOVE_X_POS = "move_x_positive"
    MOVE_X_NEG = "move_x_negative"  
    MOVE_Y_POS = "move_y_positive"
    MOVE_Y_NEG = "move_y_negative"
    MOVE_Z_POS = "move_z_positive"
    MOVE_Z_NEG = "move_z_negative"
    HOME = "home"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class RobotState:
    """Current robot state"""
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    is_moving: bool
    last_command: str
    timestamp: float
    safety_status: str

class RobotController:
    """Robot controller for EEG-based control"""
    
    def __init__(self, config_path: str = "config/robot.yaml"):
        """Initialize robot controller with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Robot state
        self.current_state = RobotState(
            position=(0.0, 0.0, 0.0),
            velocity=(0.0, 0.0, 0.0),
            is_moving=False,
            last_command="stop",
            timestamp=time.time(),
            safety_status="safe"
        )
        
        # Safety parameters
        safety_config = self.config['safety']
        self.position_limits = safety_config['position_limits']
        self.velocity_limits = safety_config['velocity_limits']
        self.max_acceleration = safety_config['max_acceleration']
        self.emergency_stop_enabled = True
        
        # Movement parameters
        movement_config = self.config['movement']
        self.step_size = movement_config['step_size']
        self.movement_speed = movement_config['speed']
        self.smoothing_enabled = movement_config['smoothing']['enabled']
        self.smoothing_factor = movement_config['smoothing']['factor']
        
        # Command queue and threading
        self.command_queue = queue.Queue(maxsize=100)
        self.control_thread = None
        self.is_running = False
        
        # Command history for smoothing
        self.command_history = []
        self.max_history_length = 10
        
        # Output configuration
        self.simulation_mode = self.config.get('simulation', {}).get('enabled', True)
        self.output_file = self.config.get('simulation', {}).get('output_file', 'robot_commands.json')
        
        # Real robot connection (placeholder for actual implementation)
        self.robot_connected = False
        
        self.logger.info(f"Robot controller initialized - Simulation mode: {self.simulation_mode}")
    
    def start(self):
        """Start the robot control system"""
        if self.is_running:
            self.logger.warning("Robot controller already running")
            return
        
        self.is_running = True
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        
        # Initialize robot position
        self._initialize_robot()
        
        self.logger.info("Robot controller started")
    
    def stop(self):
        """Stop the robot control system"""
        if not self.is_running:
            return
        
        # Send emergency stop
        self.emergency_stop()
        
        # Stop control loop
        self.is_running = False
        
        if self.control_thread:
            self.control_thread.join(timeout=5)
        
        self.logger.info("Robot controller stopped")
    
    def send_command(self, command: Union[str, RobotCommand], parameters: Dict[str, Any] = None):
        """Send command to robot"""
        if isinstance(command, str):
            try:
                command = RobotCommand(command)
            except ValueError:
                self.logger.error(f"Invalid command: {command}")
                return False
        
        if parameters is None:
            parameters = {}
        
        # Add timestamp
        command_data = {
            'command': command.value,
            'parameters': parameters,
            'timestamp': time.time()
        }
        
        try:
            self.command_queue.put(command_data, timeout=1)
            return True
        except queue.Full:
            self.logger.warning("Command queue full, dropping command")
            return False
    
    def process_eeg_prediction(self, prediction: int, confidence: float = 1.0):
        """
        Process EEG model prediction and convert to robot command
        
        Args:
            prediction: Integer prediction from EEG model (0-6 for 7 classes)
            confidence: Confidence score (0-1)
        """
        # Define mapping from prediction to robot command
        prediction_map = {
            0: RobotCommand.STOP,
            1: RobotCommand.MOVE_X_POS,
            2: RobotCommand.MOVE_X_NEG,
            3: RobotCommand.MOVE_Y_POS,
            4: RobotCommand.MOVE_Y_NEG,
            5: RobotCommand.MOVE_Z_POS,
            6: RobotCommand.MOVE_Z_NEG
        }
        
        if prediction not in prediction_map:
            self.logger.warning(f"Invalid prediction: {prediction}")
            return False
        
        command = prediction_map[prediction]
        
        # Apply confidence threshold
        confidence_threshold = self.config.get('control', {}).get('confidence_threshold', 0.7)
        if confidence < confidence_threshold:
            self.logger.debug(f"Low confidence prediction: {confidence:.3f} < {confidence_threshold}")
            command = RobotCommand.STOP
        
        # Apply smoothing if enabled
        if self.smoothing_enabled:
            command = self._apply_command_smoothing(command, confidence)
        
        # Send command
        parameters = {
            'confidence': confidence,
            'original_prediction': prediction
        }
        
        return self.send_command(command, parameters)
    
    def _apply_command_smoothing(self, command: RobotCommand, confidence: float) -> RobotCommand:
        """Apply temporal smoothing to commands"""
        # Add to history
        self.command_history.append({
            'command': command,
            'confidence': confidence,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.command_history) > self.max_history_length:
            self.command_history.pop(0)
        
        # If not enough history, return original command
        if len(self.command_history) < 3:
            return command
        
        # Count command occurrences in recent history
        recent_commands = [h['command'] for h in self.command_history[-5:]]
        command_counts = {}
        
        for cmd in recent_commands:
            command_counts[cmd] = command_counts.get(cmd, 0) + 1
        
        # Use most frequent command if it appears more than smoothing factor
        max_count = max(command_counts.values())
        smoothing_threshold = max(2, int(len(recent_commands) * self.smoothing_factor))
        
        if max_count >= smoothing_threshold:
            smoothed_command = max(command_counts, key=command_counts.get)
            return smoothed_command
        
        return command
    
    def _control_loop(self):
        """Main control loop running in separate thread"""
        self.logger.info("Robot control loop started")
        
        while self.is_running:
            try:
                # Get command from queue with timeout
                try:
                    command_data = self.command_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Process command
                self._execute_command(command_data)
                
                # Mark task as done
                self.command_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Control loop error: {e}")
                time.sleep(0.1)
        
        self.logger.info("Robot control loop stopped")
    
    def _execute_command(self, command_data: Dict[str, Any]):
        """Execute a robot command"""
        command_str = command_data['command']
        parameters = command_data.get('parameters', {})
        timestamp = command_data['timestamp']
        
        try:
            command = RobotCommand(command_str)
        except ValueError:
            self.logger.error(f"Invalid command in execution: {command_str}")
            return
        
        # Check safety constraints
        if not self._check_safety_constraints(command):
            self.logger.warning(f"Safety constraint violation for command: {command.value}")
            self._execute_emergency_stop()
            return
        
        # Log command execution
        confidence = parameters.get('confidence', 1.0)
        self.logger.info(f"Executing command: {command.value} (confidence: {confidence:.3f})")
        
        # Update state
        self.current_state.last_command = command.value
        self.current_state.timestamp = timestamp
        
        # Execute command based on type
        if command == RobotCommand.STOP:
            self._execute_stop()
        elif command == RobotCommand.HOME:
            self._execute_home()
        elif command == RobotCommand.EMERGENCY_STOP:
            self._execute_emergency_stop()
        else:
            self._execute_movement(command, parameters)
        
        # Output command for simulation/logging
        self._output_command(command_data)
    
    def _execute_movement(self, command: RobotCommand, parameters: Dict[str, Any]):
        """Execute movement command"""
        current_pos = list(self.current_state.position)
        step = self.step_size
        
        # Apply confidence scaling
        confidence = parameters.get('confidence', 1.0)
        scaled_step = step * confidence
        
        # Calculate new position based on command
        if command == RobotCommand.MOVE_X_POS:
            current_pos[0] += scaled_step
        elif command == RobotCommand.MOVE_X_NEG:
            current_pos[0] -= scaled_step
        elif command == RobotCommand.MOVE_Y_POS:
            current_pos[1] += scaled_step
        elif command == RobotCommand.MOVE_Y_NEG:
            current_pos[1] -= scaled_step
        elif command == RobotCommand.MOVE_Z_POS:
            current_pos[2] += scaled_step
        elif command == RobotCommand.MOVE_Z_NEG:
            current_pos[2] -= scaled_step
        
        # Apply position limits
        current_pos = self._apply_position_limits(current_pos)
        
        # Update state
        self.current_state.position = tuple(current_pos)
        self.current_state.is_moving = True
        
        # Calculate velocity (simplified)
        prev_pos = self.current_state.position
        dt = 0.1  # Assume 100ms update rate
        velocity = [(current_pos[i] - prev_pos[i]) / dt for i in range(3)]
        self.current_state.velocity = tuple(velocity)
    
    def _execute_stop(self):
        """Execute stop command"""
        self.current_state.is_moving = False
        self.current_state.velocity = (0.0, 0.0, 0.0)
    
    def _execute_home(self):
        """Execute home position command"""
        home_position = self.config['movement']['home_position']
        self.current_state.position = tuple(home_position)
        self.current_state.is_moving = False
        self.current_state.velocity = (0.0, 0.0, 0.0)
        self.logger.info(f"Robot moved to home position: {home_position}")
    
    def _execute_emergency_stop(self):
        """Execute emergency stop"""
        self.current_state.is_moving = False
        self.current_state.velocity = (0.0, 0.0, 0.0)
        self.current_state.safety_status = "emergency_stop"
        self.logger.warning("EMERGENCY STOP ACTIVATED")
    
    def _check_safety_constraints(self, command: RobotCommand) -> bool:
        """Check if command violates safety constraints"""
        if not self.emergency_stop_enabled:
            return False
        
        if self.current_state.safety_status == "emergency_stop":
            return command == RobotCommand.EMERGENCY_STOP
        
        # Check position limits for movement commands
        if command in [RobotCommand.MOVE_X_POS, RobotCommand.MOVE_X_NEG,
                      RobotCommand.MOVE_Y_POS, RobotCommand.MOVE_Y_NEG,
                      RobotCommand.MOVE_Z_POS, RobotCommand.MOVE_Z_NEG]:
            
            # Simulate future position
            future_pos = list(self.current_state.position)
            step = self.step_size
            
            if command == RobotCommand.MOVE_X_POS:
                future_pos[0] += step
            elif command == RobotCommand.MOVE_X_NEG:
                future_pos[0] -= step
            elif command == RobotCommand.MOVE_Y_POS:
                future_pos[1] += step
            elif command == RobotCommand.MOVE_Y_NEG:
                future_pos[1] -= step
            elif command == RobotCommand.MOVE_Z_POS:
                future_pos[2] += step
            elif command == RobotCommand.MOVE_Z_NEG:
                future_pos[2] -= step
            
            # Check if future position is within limits
            for i, (min_limit, max_limit) in enumerate(self.position_limits):
                if future_pos[i] < min_limit or future_pos[i] > max_limit:
                    return False
        
        return True
    
    def _apply_position_limits(self, position: List[float]) -> List[float]:
        """Apply position limits to prevent out-of-bounds movement"""
        limited_position = position.copy()
        
        for i, (min_limit, max_limit) in enumerate(self.position_limits):
            limited_position[i] = max(min_limit, min(max_limit, limited_position[i]))
        
        return limited_position
    
    def _initialize_robot(self):
        """Initialize robot to home position"""
        if self.simulation_mode:
            home_position = self.config['movement']['home_position']
            self.current_state.position = tuple(home_position)
            self.current_state.safety_status = "safe"
            self.logger.info(f"Robot initialized at home position: {home_position}")
        else:
            # Real robot initialization would go here
            self.logger.info("Real robot initialization not implemented")
    
    def _output_command(self, command_data: Dict[str, Any]):
        """Output command for simulation or logging"""
        if self.simulation_mode:
            # Prepare output data
            output_data = {
                'timestamp': command_data['timestamp'],
                'command': command_data['command'],
                'parameters': command_data.get('parameters', {}),
                'robot_state': {
                    'position': self.current_state.position,
                    'velocity': self.current_state.velocity,
                    'is_moving': self.current_state.is_moving,
                    'safety_status': self.current_state.safety_status
                }
            }
            
            # Write to file
            try:
                with open(self.output_file, 'a') as f:
                    f.write(json.dumps(output_data) + '\n')
            except Exception as e:
                self.logger.error(f"Failed to write command output: {e}")
    
    def get_state(self) -> RobotState:
        """Get current robot state"""
        return self.current_state
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed robot status"""
        return {
            'connected': self.robot_connected,
            'simulation_mode': self.simulation_mode,
            'is_running': self.is_running,
            'position': self.current_state.position,
            'velocity': self.current_state.velocity,
            'is_moving': self.current_state.is_moving,
            'last_command': self.current_state.last_command,
            'safety_status': self.current_state.safety_status,
            'command_queue_size': self.command_queue.qsize(),
            'emergency_stop_enabled': self.emergency_stop_enabled
        }
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        self.send_command(RobotCommand.EMERGENCY_STOP)
    
    def reset_emergency_stop(self):
        """Reset emergency stop status"""
        if self.current_state.safety_status == "emergency_stop":
            self.current_state.safety_status = "safe"
            self.logger.info("Emergency stop reset")
    
    def clear_command_queue(self):
        """Clear all pending commands"""
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
                self.command_queue.task_done()
            except queue.Empty:
                break
        self.logger.info("Command queue cleared")

def main():
    """Test the robot controller"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Robot Controller")
    parser.add_argument('--config', default='config/robot.yaml', help='Config file path')
    parser.add_argument('--duration', type=int, default=30, help='Test duration (seconds)')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = RobotController(args.config)
    
    try:
        # Start controller
        print("Starting robot controller...")
        controller.start()
        
        print(f"Initial status: {controller.get_status()}")
        
        if args.interactive:
            print("\nInteractive mode - Available commands:")
            print("0: Stop, 1: X+, 2: X-, 3: Y+, 4: Y-, 5: Z+, 6: Z-, 7: Home, 8: Emergency Stop")
            print("Enter command number (or 'q' to quit):")
            
            while True:
                try:
                    user_input = input("> ").strip()
                    
                    if user_input.lower() == 'q':
                        break
                    
                    prediction = int(user_input)
                    confidence = 0.9  # High confidence for manual commands
                    
                    if prediction == 7:
                        controller.send_command(RobotCommand.HOME)
                    elif prediction == 8:
                        controller.emergency_stop()
                    else:
                        controller.process_eeg_prediction(prediction, confidence)
                    
                    # Show status
                    status = controller.get_status()
                    print(f"Position: {status['position']}, Moving: {status['is_moving']}")
                    
                except (ValueError, KeyboardInterrupt):
                    break
        
        else:
            # Automated test sequence
            print(f"Running automated test for {args.duration} seconds...")
            
            test_commands = [
                (1, 0.8),  # X+
                (0, 1.0),  # Stop
                (3, 0.9),  # Y+
                (0, 1.0),  # Stop
                (5, 0.7),  # Z+
                (0, 1.0),  # Stop
                (2, 0.8),  # X-
                (4, 0.9),  # Y-
                (6, 0.7),  # Z-
            ]
            
            start_time = time.time()
            command_idx = 0
            
            while time.time() - start_time < args.duration:
                if command_idx < len(test_commands):
                    prediction, confidence = test_commands[command_idx]
                    controller.process_eeg_prediction(prediction, confidence)
                    command_idx += 1
                    
                    print(f"Sent command {prediction} with confidence {confidence}")
                    status = controller.get_status()
                    print(f"Robot status: Position={status['position']}, Moving={status['is_moving']}")
                
                time.sleep(2)
            
            # Return to home
            print("Returning to home position...")
            controller.send_command(RobotCommand.HOME)
            time.sleep(2)
        
        # Final status
        print(f"\nFinal status: {controller.get_status()}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("Stopping robot controller...")
        controller.stop()
        print("Done!")

if __name__ == "__main__":
    main()
