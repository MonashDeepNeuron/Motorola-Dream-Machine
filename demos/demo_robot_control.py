#!/usr/bin/env python3
"""
Demo 4: Robot Command Generation with Live JSONL Streaming
=========================================================

This demonstration shows how predictions become robot movements with continuous
JSONL streaming to asynchronous_deltas.jsonl (like your current file).

Usage:
    python3 demos/demo_robot_control.py

Output:
    - 3D robot position tracking
    - Safety limit enforcement  
    - Command execution timing
    - Live JSONL streaming to output/robot_commands.jsonl
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.robot.controller import RobotController
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class LiveJSONLStreamer:
    """Streams robot commands to JSONL file in real-time."""
    
    def __init__(self, output_file="output/robot_commands.jsonl"):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file with header comment
        with open(self.output_file, 'w') as f:
            f.write(f'# Robot Commands JSONL Stream - Started {datetime.now().isoformat()}\n')
    
    def log_command(self, command, confidence, position, delta):
        """Log a robot command to JSONL file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "command": command,
            "confidence": float(confidence),
            "position": {
                "x": float(position[0]),
                "y": float(position[1]), 
                "z": float(position[2])
            },
            "dx": float(delta[0]),
            "dy": float(delta[1]),
            "dz": float(delta[2])
        }
        
        # Append to JSONL file (each line is valid JSON)
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            f.flush()  # Ensure immediate write

def simulate_brain_commands():
    """Simulate brain command predictions with realistic patterns."""
    commands = [
        'move_x_positive', 'move_x_negative',
        'move_y_positive', 'move_y_negative', 
        'move_z_positive', 'move_z_negative',
        'stop'
    ]
    
    # Simulate realistic command sequence
    command_sequence = [
        ('move_y_positive', 0.85),   # Forward
        ('move_y_positive', 0.78),   # Continue forward
        ('stop', 0.92),              # Stop
        ('move_x_positive', 0.81),   # Right
        ('move_z_positive', 0.87),   # Up
        ('stop', 0.89),              # Stop
        ('move_x_negative', 0.76),   # Left
        ('move_y_negative', 0.83),   # Backward
        ('move_z_negative', 0.79),   # Down
        ('stop', 0.95),              # Final stop
    ]
    
    for command, confidence in command_sequence:
        yield command, confidence
        time.sleep(0.5)  # Simulate real-time delay

def demo_robot_control():
    """Demonstrate robot control with live JSONL streaming."""
    print("🤖 Demo 4: Robot Command Generation")
    print("=" * 50)
    
    # Setup
    logger = setup_logging(log_level="INFO")
    
    # Initialize robot controller
    print("🦾 Initializing robot controller...")
    robot = RobotController()
    
    # Initialize JSONL streamer
    print("📝 Setting up live JSONL streaming...")
    streamer = LiveJSONLStreamer("output/robot_commands.jsonl")
    
    # Also stream to the user's existing file format
    legacy_streamer = LiveJSONLStreamer("ursim_test_v1/asynchronous_deltas.jsonl")
    
    print(f"   - Streaming to: {streamer.output_file}")
    print(f"   - Legacy format: {legacy_streamer.output_file}")
    
    # Setup tracking
    positions = []
    timestamps = []
    commands_executed = []
    
    print("\n🧠 Starting brain-controlled robot simulation...")
    print("   Commands will be executed every 0.5 seconds")
    print("   Watch the JSONL files for live updates!")
    print("   Press Ctrl+C to stop\n")
    
    try:
        start_time = time.time()
        
        for i, (command, confidence) in enumerate(simulate_brain_commands()):
            current_time = time.time() - start_time
            
            print(f"⚡ [{current_time:6.1f}s] Brain Command: {command:15} (confidence: {confidence:.3f})")
            
            # Execute command on robot
            success = robot.send_command(command, {'confidence': confidence})
            
            if success:
                # Get current position from robot
                robot_status = robot.get_status()
                current_pos = robot_status['position']
                
                # Calculate delta based on command
                command_deltas = {
                    'move_x_positive': [0.05, 0.0, 0.0],
                    'move_x_negative': [-0.05, 0.0, 0.0], 
                    'move_y_positive': [0.0, 0.05, 0.0],
                    'move_y_negative': [0.0, -0.05, 0.0],
                    'move_z_positive': [0.0, 0.0, 0.05],
                    'move_z_negative': [0.0, 0.0, -0.05],
                    'stop': [0.0, 0.0, 0.0]
                }
                
                delta = command_deltas.get(command, [0.0, 0.0, 0.0])
                position = [current_pos[0], current_pos[1], current_pos[2]]
                
                print(f"   ✅ Robot moved to: [{position[0]:6.3f}, {position[1]:6.3f}, {position[2]:6.3f}]")
                print(f"   📊 Delta: [{delta[0]:6.3f}, {delta[1]:6.3f}, {delta[2]:6.3f}]")
                
                # Stream to JSONL files
                streamer.log_command(command, confidence, position, delta)
                
                # Also log in the legacy format (matching your asynchronous_deltas.jsonl)
                legacy_entry = {
                    "dx": float(delta[0]),
                    "dy": float(delta[1]), 
                    "dz": float(delta[2])
                }
                with open("ursim_test_v1/asynchronous_deltas.jsonl", 'a') as f:
                    f.write(json.dumps(legacy_entry) + '\n')
                
                # Track for visualization
                positions.append(position.copy())
                timestamps.append(current_time)
                commands_executed.append(command)
                
            else:
                print(f"   ❌ Command failed: Robot command unsuccessful")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    
    print(f"\n📊 Demo Summary:")
    print(f"   - Commands executed: {len(positions)}")
    print(f"   - Total runtime: {timestamps[-1] if timestamps else 0:.1f} seconds")
    print(f"   - Final position: {positions[-1] if positions else 'Unknown'}")
    
    # Create visualizations
    if positions:
        print("\n📈 Creating movement visualization...")
        
        output_dir = Path("output/demo4_robot_control")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        positions = np.array(positions)
        
        # 3D trajectory plot
        fig = plt.figure(figsize=(15, 5))
        
        # 3D plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-o', markersize=4)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_zlabel('Z Position (m)')
        ax1.set_title('3D Robot Trajectory')
        
        # X-Y trajectory
        ax2 = fig.add_subplot(132)
        ax2.plot(positions[:, 0], positions[:, 1], 'r-o', markersize=4)
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)') 
        ax2.set_title('Top View (X-Y Plane)')
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
        
        # Position vs time
        ax3 = fig.add_subplot(133)
        ax3.plot(timestamps, positions[:, 0], 'r-', label='X')
        ax3.plot(timestamps, positions[:, 1], 'g-', label='Y')
        ax3.plot(timestamps, positions[:, 2], 'b-', label='Z')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position (m)')
        ax3.set_title('Position vs Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'robot_trajectory.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Command distribution
        plt.figure(figsize=(10, 6))
        unique_commands, counts = np.unique(commands_executed, return_counts=True)
        plt.bar(unique_commands, counts, color='skyblue', edgecolor='navy')
        plt.xlabel('Robot Commands')
        plt.ylabel('Frequency')
        plt.title('Command Execution Distribution')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'command_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Show JSONL file contents
    print(f"\n📄 JSONL Stream Contents (last 5 entries):")
    if streamer.output_file.exists():
        with open(streamer.output_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:
                if line.strip() and not line.startswith('#'):
                    try:
                        data = json.loads(line)
                        print(f"   {data}")
                    except json.JSONDecodeError:
                        pass
    
    print(f"\n✅ Demo completed!")
    print(f"📁 Outputs saved to: output/demo4_robot_control/")
    print(f"📝 Live JSONL stream: {streamer.output_file}")
    print(f"📝 Legacy format: ursim_test_v1/asynchronous_deltas.jsonl")
    print("\n📋 Key Features Demonstrated:")
    print("   - Real-time robot command execution")
    print("   - Safety limit enforcement")
    print("   - Continuous JSONL streaming")
    print("   - 3D position tracking")
    print("   - Movement visualization")
    print("   - Command timing analysis")

if __name__ == "__main__":
    try:
        demo_robot_control()
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()
