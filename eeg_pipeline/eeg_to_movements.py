#!/usr/bin/env python3
"""
EEG → Movement Deltas

Reads EEG file, performs real-time band power analysis,
outputs movement deltas (dx, dy, dz, drx, dry, drz) to JSONL file.

Compatible with your ur_asynchronous.py robot controller.

Usage:
    # Terminal 1: Start robot (in ursim_test_v1/)
    python ur_asynchronous.py --robot-ip 127.0.0.1 --json-file ../eeg_pipeline/movements.jsonl
    
    # Terminal 2: Generate movements from EEG (in eeg_pipeline/)
    python eeg_to_movements.py --edf-file "path/to/emotiv.edf" --output movements.jsonl
"""

import argparse
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import mne
import numpy as np
from scipy import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EEGToMovement:
    """Convert EEG band powers to robot movement deltas"""
    
    def __init__(self, velocity_scale: float = 0.05, enable_limits: bool = True):
        """
        Args:
            velocity_scale: Scale factor for velocities (m/s)
                           Default 0.05 m/s = 50 mm/s
            enable_limits: Enable safety position limits
        """
        self.velocity_scale = velocity_scale
        self.enable_limits = enable_limits
        
        # Safety: Track accumulated position to prevent hitting limits
        self.position = {'x': 0.0, 'y': 0.0, 'z': 0.3}  # Start at safe position
        self.position_limits = {
            'x': (-0.5, 0.5),   # ±500mm
            'y': (-0.5, 0.5),   # ±500mm  
            'z': (0.1, 0.8)     # 100mm to 800mm height
        }
        
        # Band frequency definitions
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        self.command_count = 0
        
        logger.info(f"EEG → Movement converter initialized")
        logger.info(f"Velocity scale: {velocity_scale} m/s ({velocity_scale*1000} mm/s)")
        if enable_limits:
            logger.info(f"Safety limits enabled: X={self.position_limits['x']}, "
                       f"Y={self.position_limits['y']}, Z={self.position_limits['z']}")
    
    def compute_band_powers(
        self,
        data: np.ndarray,
        sfreq: float,
        channel_names: list
    ) -> dict:
        """
        Compute average band powers across EEG channels.
        
        Args:
            data: EEG data (channels × samples)
            sfreq: Sampling frequency
            channel_names: Channel names
            
        Returns:
            Dict of band → power
        """
        # Filter to EEG channels only
        eeg_channels = []
        for i, ch in enumerate(channel_names):
            # Exclude Emotiv metadata channels
            if not any(x in ch for x in ['TIME_STAMP', 'COUNTER', 'INTERPOLATED', 'OR_']):
                eeg_channels.append(i)
        
        if not eeg_channels:
            return {}
        
        eeg_data = data[eeg_channels, :]
        
        # Compute PSD using Welch
        nperseg = min(int(sfreq), eeg_data.shape[1])
        freqs, psd = signal.welch(
            eeg_data,
            fs=sfreq,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            axis=-1
        )
        
        # Average across channels
        avg_psd = np.mean(psd, axis=0)
        
        # Extract band powers
        band_powers = {}
        for band_name, (low, high) in self.bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            if np.any(idx):
                band_powers[band_name] = np.mean(avg_psd[idx])
            else:
                band_powers[band_name] = 0.0
        
        return band_powers
    
    def band_to_velocity(self, band_powers: dict) -> tuple:
        """
        Convert band powers to velocity deltas with MULTI-AXIS control.
        
        Enhanced mapping using band power ratios:
        - Delta (0.5-4 Hz): Deep rest → Move backward (negative X)
        - Theta (4-8 Hz): Drowsy → Move down (negative Z)
        - Alpha (8-13 Hz): Relaxed → Move forward (positive X)
        - Beta (13-30 Hz): Focused → Move up (positive Z)
        - Gamma (30-45 Hz): Alert → Move right (positive Y)
        
        Multi-axis: Combines secondary bands for diagonal/complex movements
        
        Returns:
            (dx, dy, dz, drx, dry, drz): Velocity deltas in m/s
        """
        if not band_powers:
            return (0, 0, 0, 0, 0, 0)
        
        # Normalize band powers
        total_power = sum(band_powers.values())
        if total_power == 0:
            return (0, 0, 0, 0, 0, 0)
        
        normalized = {k: v / total_power for k, v in band_powers.items()}
        
        # Find dominant and secondary bands
        sorted_bands = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        dominant = sorted_bands[0][0]
        dominant_ratio = sorted_bands[0][1]
        
        # Calculate base velocity from dominant band strength
        # If band has >15% of total power, it's strong enough
        strength = min((dominant_ratio - 0.1) * 5, 1.0)  # Scale 0.1-0.3 → 0-1
        strength = max(strength, 0.0)
        
        base_vel = self.velocity_scale * strength
        
        # Initialize velocities
        dx, dy, dz = 0.0, 0.0, 0.0
        
        # Axis-specific scaling factors for more noticeable dy/dz movement
        # X-axis (forward/back) is common, so keep it at 1.0x
        # Y-axis (left/right) needs 4x boost for visibility
        # Z-axis (up/down) needs 3x boost for visibility
        y_scale = 4.0
        z_scale = 3.0
        
        # Primary movement from dominant band
        primary_movements = {
            'delta': (-base_vel, 0, 0),              # Backward
            'theta': (0, 0, -base_vel * z_scale),    # Down (3x boost)
            'alpha': (base_vel, 0, 0),               # Forward
            'beta': (0, 0, base_vel * z_scale),      # Up (3x boost)
            'gamma': (0, base_vel * y_scale, 0),     # Right (4x boost)
        }
        
        dx, dy, dz = primary_movements.get(dominant, (0, 0, 0))
        
        # Add secondary movement if second band is strong (>10% threshold - LOWERED for more multi-axis)
        if len(sorted_bands) > 1 and sorted_bands[1][1] > 0.10:
            secondary = sorted_bands[1][0]
            secondary_ratio = sorted_bands[1][1]
            secondary_vel = self.velocity_scale * secondary_ratio * 0.6  # 60% strength for secondary
            
            secondary_movements = {
                'delta': (-secondary_vel, 0, 0),
                'theta': (0, 0, -secondary_vel * z_scale),
                'alpha': (secondary_vel, 0, 0),
                'beta': (0, 0, secondary_vel * z_scale),
                'gamma': (0, secondary_vel * y_scale, 0),
            }
            
            sdx, sdy, sdz = secondary_movements.get(secondary, (0, 0, 0))
            dx += sdx
            dy += sdy
            dz += sdz
        
        # Add tertiary movement if third band is present (>8% threshold - for richer 3D motion)
        if len(sorted_bands) > 2 and sorted_bands[2][1] > 0.08:
            tertiary = sorted_bands[2][0]
            tertiary_ratio = sorted_bands[2][1]
            tertiary_vel = self.velocity_scale * tertiary_ratio * 0.4  # 40% strength
            
            tertiary_movements = {
                'delta': (-tertiary_vel, 0, 0),
                'theta': (0, 0, -tertiary_vel * z_scale),
                'alpha': (tertiary_vel, 0, 0),
                'beta': (0, 0, tertiary_vel * z_scale),
                'gamma': (0, tertiary_vel * y_scale, 0),
            }
            
            tdx, tdy, tdz = tertiary_movements.get(tertiary, (0, 0, 0))
            dx += tdx
            dy += tdy
            dz += tdz
        
        # Safety: Check if movement would exceed limits
        if self.enable_limits:
            # Predict next position (assuming 0.5s execution time)
            dt = 0.5
            next_x = self.position['x'] + dx * dt
            next_y = self.position['y'] + dy * dt
            next_z = self.position['z'] + dz * dt
            
            # Clamp velocities if approaching limits
            if next_x <= self.position_limits['x'][0] or next_x >= self.position_limits['x'][1]:
                logger.warning(f"⚠️  X limit approaching ({next_x:.2f}m), stopping X motion")
                dx = 0
            if next_y <= self.position_limits['y'][0] or next_y >= self.position_limits['y'][1]:
                logger.warning(f"⚠️  Y limit approaching ({next_y:.2f}m), stopping Y motion")
                dy = 0
            if next_z <= self.position_limits['z'][0] or next_z >= self.position_limits['z'][1]:
                logger.warning(f"⚠️  Z limit approaching ({next_z:.2f}m), stopping Z motion")
                dz = 0
            
            # Update tracked position
            self.position['x'] += dx * dt
            self.position['y'] += dy * dt
            self.position['z'] += dz * dt
        
        # No rotation for now
        drx, dry, drz = 0, 0, 0
        
        return (dx, dy, dz, drx, dry, drz)
    
    def create_movement_command(self, band_powers: dict) -> dict:
        """
        Create movement command dict compatible with ur_asynchronous.py
        
        Returns:
            dict with keys: dx, dy, dz, drx, dry, drz
        """
        self.command_count += 1
        
        dx, dy, dz, drx, dry, drz = self.band_to_velocity(band_powers)
        
        # Log every 10th command
        if self.command_count % 10 == 0:
            if band_powers:
                # Show top 3 bands
                sorted_bands = sorted(band_powers.items(), key=lambda x: x[1], reverse=True)
                total = sum(band_powers.values())
                
                bands_parts = []
                for i, (band, power) in enumerate(sorted_bands[:3]):
                    pct = power / total * 100
                    if i == 0 or pct > 8:  # Show if dominant or >8%
                        bands_parts.append(f"{band}({pct:.0f}%)")
                
                bands_str = " + ".join(bands_parts) if bands_parts else sorted_bands[0][0]
            else:
                bands_str = "none"
            
            if self.enable_limits:
                logger.info(
                    f"Cmd #{self.command_count}: {bands_str} → "
                    f"vel=({dx:.3f}, {dy:.3f}, {dz:.3f}) m/s | "
                    f"pos=({self.position['x']:.2f}, {self.position['y']:.2f}, {self.position['z']:.2f})m"
                )
            else:
                logger.info(
                    f"Cmd #{self.command_count}: {bands_str} → "
                    f"vel=({dx:.3f}, {dy:.3f}, {dz:.3f}) m/s"
                )
        
        return {
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'drx': drx,
            'dry': dry,
            'drz': drz,
            'timestamp': datetime.now().isoformat(),
            'command_number': self.command_count,
            'dominant_band': max(band_powers.items(), key=lambda x: x[1])[0] if band_powers else "none"
        }


def process_eeg_to_movements(
    edf_file: str,
    output_file: str,
    window_size: float = 2.0,
    overlap: float = 0.5,
    speed: float = 1.0,
    velocity_scale: float = 0.05
):
    """
    Process EEG file and generate movement commands.
    
    Args:
        edf_file: Path to EDF file
        output_file: Output JSONL file
        window_size: Analysis window (seconds)
        overlap: Window overlap (0-1)
        speed: Playback speed multiplier
        velocity_scale: Velocity scale (m/s)
    """
    logger.info("=" * 60)
    logger.info("EEG → Movement Delta Generator")
    logger.info("=" * 60)
    logger.info(f"Input: {edf_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Window: {window_size}s, Overlap: {overlap*100}%")
    logger.info(f"Speed: {speed}x")
    logger.info("=" * 60)
    
    # Load EEG
    logger.info("Loading EEG data...")
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    channel_names = raw.ch_names
    
    logger.info(f"Loaded: {len(channel_names)} channels @ {sfreq} Hz")
    logger.info(f"Duration: {raw.times[-1]:.1f}s")
    
    # Initialize converter
    converter = EEGToMovement(velocity_scale=velocity_scale)
    
    # Window parameters
    window_samples = int(window_size * sfreq)
    step_samples = int(window_samples * (1 - overlap))
    
    logger.info(f"Processing {window_samples} samples per window")
    logger.info(f"Step size: {step_samples} samples")
    
    # Prepare output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we should append or overwrite
    if output_path.exists():
        logger.info(f"⚠️  File exists, will APPEND to: {output_file}")
        logger.info("")
    else:
        # Create new file
        with open(output_file, 'w') as f:
            f.write('')
        logger.info(f"📝 Created new file: {output_file}")
        logger.info("")
    
    logger.info("🎬 Processing... (Robot can read file in real-time)")
    logger.info("")
    
    # Get data
    data, times = raw.get_data(return_times=True)
    n_samples = data.shape[1]
    
    # Process windows
    window_count = 0
    start_time = time.time()
    
    for start_idx in range(0, n_samples - window_samples, step_samples):
        end_idx = start_idx + window_samples
        
        # Extract window
        window_data = data[:, start_idx:end_idx]
        window_time = times[start_idx]
        
        # Compute band powers
        band_powers = converter.compute_band_powers(
            window_data,
            sfreq,
            channel_names
        )
        
        # Create movement command
        command = converter.create_movement_command(band_powers)
        
        # Append to file (robot reads this asynchronously)
        with open(output_file, 'a') as f:
            f.write(json.dumps(command) + '\n')
        
        window_count += 1
        
        # Show progress
        if window_count % 20 == 0:
            logger.info(f"Window {window_count} @ {window_time:.1f}s EEG time")
        
        # Simulate real-time
        sleep_time = (step_samples / sfreq) / speed
        time.sleep(sleep_time)
    
    elapsed = time.time() - start_time
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ Complete!")
    logger.info("=" * 60)
    logger.info(f"Windows: {window_count}")
    logger.info(f"Commands: {converter.command_count}")
    logger.info(f"Time: {elapsed:.1f}s")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert EEG to movement deltas for robot control"
    )
    parser.add_argument(
        '--edf-file',
        required=True,
        help='EDF file path'
    )
    parser.add_argument(
        '--output',
        default='movements.jsonl',
        help='Output JSONL file (default: movements.jsonl)'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=2.0,
        help='Window size in seconds (default: 2.0)'
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Playback speed (default: 1.0)'
    )
    parser.add_argument(
        '--velocity-scale',
        type=float,
        default=0.05,
        help='Velocity scale in m/s (default: 0.05 = 50mm/s)'
    )
    
    args = parser.parse_args()
    
    # Check file exists
    if not Path(args.edf_file).exists():
        logger.error(f"File not found: {args.edf_file}")
        return 1
    
    try:
        process_eeg_to_movements(
            edf_file=args.edf_file,
            output_file=args.output,
            window_size=args.window_size,
            speed=args.speed,
            velocity_scale=args.velocity_scale
        )
        return 0
    except KeyboardInterrupt:
        logger.info("\n⏸️  Stopped")
        return 0
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit(main())
