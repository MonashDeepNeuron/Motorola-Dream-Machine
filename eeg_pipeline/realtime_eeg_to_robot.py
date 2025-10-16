#!/usr/bin/env python3
"""
Real-Time EEG → Robot Control (No Kafka Required)

Reads EEG data from file, performs real-time band power analysis,
converts to robot commands, and writes to JSON file for robot consumption.

Usage:
    python realtime_eeg_to_robot.py --edf-file <path> --output robot_commands.json
    
Then in another terminal:
    python integrated_robot_controller.py --robot-type mock --file robot_commands.json
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import deque

import mne
import numpy as np
from scipy import signal

from schemas.robot_schemas import RobotCommand

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeEEGProcessor:
    """
    Process EEG data in real-time windows and convert to robot commands.
    
    Uses sliding window approach:
    - Window size: 2 seconds (512 samples @ 256 Hz)
    - Overlap: 50% (1 second step)
    """
    
    def __init__(
        self,
        window_size: float = 2.0,
        overlap: float = 0.5,
        movement_scale: float = 20.0
    ):
        self.window_size = window_size
        self.overlap = overlap
        self.movement_scale = movement_scale
        
        # Robot state
        self.position = {'x': 0.0, 'y': 0.0, 'z': 300.0}
        self.velocity_limit = 50.0  # mm/s
        self.position_limits = {
            'x': (-500, 500),
            'y': (-500, 500),
            'z': (0, 600)
        }
        
        self.command_count = 0
        
        # Band frequency definitions
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        logger.info("Real-time EEG processor initialized")
        logger.info(f"Window: {window_size}s, Overlap: {overlap*100}%")
        logger.info(f"Movement scale: {movement_scale} mm/s")
    
    def compute_band_powers(
        self,
        data: np.ndarray,
        sfreq: float,
        channel_names: List[str]
    ) -> Dict[str, float]:
        """
        Compute average band powers across all EEG channels.
        
        Args:
            data: EEG data (channels × samples)
            sfreq: Sampling frequency
            channel_names: Channel names
            
        Returns:
            Dict of band name → average power
        """
        # Filter to only EEG channels (exclude metadata channels)
        eeg_channels = []
        for i, ch in enumerate(channel_names):
            if not any(x in ch for x in ['TIME_STAMP', 'COUNTER', 'INTERPOLATED', 'OR_']):
                eeg_channels.append(i)
        
        if not eeg_channels:
            logger.warning("No EEG channels found!")
            return {}
        
        eeg_data = data[eeg_channels, :]
        
        # Compute PSD using Welch's method
        nperseg = min(int(sfreq), eeg_data.shape[1])
        freqs, psd = signal.welch(
            eeg_data,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            axis=-1
        )
        
        # Average PSD across channels
        avg_psd = np.mean(psd, axis=0)
        
        # Compute band powers
        band_powers = {}
        for band_name, (low, high) in self.bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx):
                band_powers[band_name] = np.mean(avg_psd[idx])
            else:
                band_powers[band_name] = 0.0
        
        return band_powers
    
    def band_to_movement(self, band_powers: Dict[str, float]) -> tuple:
        """
        Convert dominant band to movement deltas.
        
        Rules:
        - Delta (0.5-4 Hz): Deep relaxation → Move back
        - Theta (4-8 Hz): Drowsy → Move down
        - Alpha (8-13 Hz): Relaxed focus → Move forward
        - Beta (13-30 Hz): Active concentration → Move up
        - Gamma (30-45 Hz): High alertness → Move right
        
        Returns:
            (dx, dy, dz): Movement deltas in mm
        """
        if not band_powers:
            return (0, 0, 0)
        
        # Find dominant band
        dominant_band = max(band_powers.items(), key=lambda x: x[1])[0]
        avg_power = sum(band_powers.values()) / len(band_powers)
        
        # Calculate strength (how much stronger is dominant band)
        strength = band_powers[dominant_band] / avg_power if avg_power > 0 else 1.0
        strength = min((strength - 1.0), 1.0)  # Normalize to 0-1
        
        # Scale movement
        move = self.movement_scale * strength
        
        # Map band to direction
        movements = {
            'delta': (-move, 0, 0),      # Back
            'theta': (0, 0, -move),      # Down
            'alpha': (move, 0, 0),       # Forward
            'beta': (0, 0, move),        # Up
            'gamma': (0, move, 0),       # Right
        }
        
        dx, dy, dz = movements.get(dominant_band, (0, 0, 0))
        
        if self.command_count % 10 == 0:
            logger.info(
                f"Dominant: {dominant_band} (strength={strength:.2f}) → "
                f"Move ({dx:.1f}, {dy:.1f}, {dz:.1f})"
            )
        
        return dx, dy, dz
    
    def update_position(self, dx: float, dy: float, dz: float, dt: float = 0.1):
        """Update robot position with velocity and position limits"""
        max_delta = self.velocity_limit * dt
        
        # Apply velocity limits
        dx = max(-max_delta, min(max_delta, dx))
        dy = max(-max_delta, min(max_delta, dy))
        dz = max(-max_delta, min(max_delta, dz))
        
        # Apply position limits
        self.position['x'] = max(
            self.position_limits['x'][0],
            min(self.position_limits['x'][1], self.position['x'] + dx)
        )
        self.position['y'] = max(
            self.position_limits['y'][0],
            min(self.position_limits['y'][1], self.position['y'] + dy)
        )
        self.position['z'] = max(
            self.position_limits['z'][0],
            min(self.position_limits['z'][1], self.position['z'] + dz)
        )
    
    def create_robot_command(self) -> dict:
        """Create robot command dict"""
        self.command_count += 1
        
        cmd = RobotCommand(
            command_type="move_to_position",
            parameters={
                "x": self.position['x'],
                "y": self.position['y'],
                "z": self.position['z'],
                "speed": 0.5
            },
            metadata={
                "control_mode": "realtime_band_power",
                "command_number": self.command_count,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        return cmd.model_dump()


def stream_edf_realtime(
    edf_file: str,
    output_file: str,
    speed: float = 1.0,
    window_size: float = 2.0,
    movement_scale: float = 20.0
):
    """
    Stream EEG data from file and generate robot commands in real-time.
    
    Args:
        edf_file: Path to EDF file
        output_file: Output JSON file for robot commands
        speed: Playback speed multiplier
        window_size: Analysis window size (seconds)
        movement_scale: Movement speed scale (mm/s)
    """
    logger.info("=" * 60)
    logger.info("Real-Time EEG → Robot Control")
    logger.info("=" * 60)
    logger.info(f"Input: {edf_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Speed: {speed}x")
    logger.info("=" * 60)
    
    # Load EDF file
    logger.info("Loading EEG data...")
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    channel_names = raw.ch_names
    n_channels = len(channel_names)
    
    logger.info(f"Loaded: {n_channels} channels @ {sfreq} Hz")
    logger.info(f"Duration: {raw.times[-1]:.1f} seconds")
    
    # Initialize processor
    processor = RealtimeEEGProcessor(
        window_size=window_size,
        overlap=0.5,
        movement_scale=movement_scale
    )
    
    # Calculate window parameters
    window_samples = int(window_size * sfreq)
    step_samples = int(window_samples * (1 - processor.overlap))
    
    logger.info(f"Window: {window_samples} samples ({window_size}s)")
    logger.info(f"Step: {step_samples} samples")
    
    # Prepare output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear output file
    with open(output_file, 'w') as f:
        f.write('')
    
    logger.info("")
    logger.info("🎬 Starting real-time processing...")
    logger.info("   (Robot can read from file in parallel)")
    logger.info("")
    
    # Get data array
    data, times = raw.get_data(return_times=True)
    n_samples = data.shape[1]
    
    # Process in sliding windows
    window_count = 0
    start_time = time.time()
    
    for start_idx in range(0, n_samples - window_samples, step_samples):
        end_idx = start_idx + window_samples
        
        # Extract window
        window_data = data[:, start_idx:end_idx]
        window_time = times[start_idx]
        
        # Compute band powers
        band_powers = processor.compute_band_powers(
            window_data,
            sfreq,
            channel_names
        )
        
        # Convert to movement
        dx, dy, dz = processor.band_to_movement(band_powers)
        processor.update_position(dx, dy, dz)
        
        # Create robot command
        command = processor.create_robot_command()
        
        # Append to file (so robot can read it)
        with open(output_file, 'a') as f:
            f.write(json.dumps(command) + '\n')
        
        window_count += 1
        
        # Show progress every 10 windows
        if window_count % 10 == 0:
            elapsed = time.time() - start_time
            eeg_time = window_time
            logger.info(
                f"Window {window_count}: EEG @ {eeg_time:.1f}s | "
                f"Position: ({processor.position['x']:.1f}, "
                f"{processor.position['y']:.1f}, {processor.position['z']:.1f})"
            )
        
        # Simulate real-time playback
        sleep_time = (step_samples / sfreq) / speed
        time.sleep(sleep_time)
    
    # Final stats
    elapsed_total = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Processing Complete!")
    logger.info("=" * 60)
    logger.info(f"Windows processed: {window_count}")
    logger.info(f"Commands generated: {processor.command_count}")
    logger.info(f"Time elapsed: {elapsed_total:.1f}s")
    logger.info(f"Final position: ({processor.position['x']:.1f}, "
                f"{processor.position['y']:.1f}, {processor.position['z']:.1f})")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Real-time EEG → Robot control (no Kafka)"
    )
    parser.add_argument(
        '--edf-file',
        required=True,
        help='Path to EDF file'
    )
    parser.add_argument(
        '--output',
        default='robot_commands.json',
        help='Output JSON file for robot commands'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Playback speed multiplier'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=2.0,
        help='Analysis window size in seconds'
    )
    parser.add_argument(
        '--movement-scale',
        type=float,
        default=20.0,
        help='Movement speed scale (mm/s)'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.edf_file).exists():
        logger.error(f"EDF file not found: {args.edf_file}")
        return 1
    
    try:
        stream_edf_realtime(
            edf_file=args.edf_file,
            output_file=args.output,
            speed=args.speed,
            window_size=args.window_size,
            movement_scale=args.movement_scale
        )
        return 0
    except KeyboardInterrupt:
        logger.info("\n⏸️  Stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
