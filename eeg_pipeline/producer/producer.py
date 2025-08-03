#!/usr/bin/env python3
"""
Simple Real-time EDF Producer
Streams EDF data as individual EEG samples in real-time.
"""

import argparse
import json
import sys
import os
import time
from uuid import uuid4
from datetime import datetime, timezone

import numpy as np
import mne
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from schemas.eeg_schemas import EEGSample


def stream_edf_realtime(edf_path: str, bootstrap_servers: str, speed_factor: float = 1.0):
    """Stream EDF data as individual samples in real-time."""
    
    # Load EDF file
    print(f"Loading EDF file: {edf_path}")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    print(f"Loaded: {len(raw.ch_names)} channels, {raw.n_times} samples, {raw.info['sfreq']} Hz")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    
    # Get data in microvolts for typical EEG range
    data_uv = raw.get_data(units="uV")  # Shape: (n_channels, n_samples)
    n_channels, n_samples = data_uv.shape
    sfreq = raw.info['sfreq']
    
    # Calculate real-time delay
    sample_interval = 1.0 / sfreq / speed_factor
    
    print(f"Streaming at {speed_factor}x speed ({sample_interval*1000:.1f}ms between samples)")
    
    # Setup Kafka producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        compression_type="lz4",
    )
    
    # Generate session info
    session_id = uuid4()
    device_id = f"edf_realtime_{os.getpid()}"
    
    print(f"Session: {session_id}")
    print(f"Device: {device_id}")
    print("Starting real-time stream... (Ctrl-C to stop)")
    
    try:
        for sample_idx in range(n_samples):
            # Get current sample from all channels
            sample_data = data_uv[:, sample_idx].tolist()
            
            # Create EEG sample message
            eeg_sample = EEGSample(
                device_id=device_id,
                session_id=session_id,
                timestamp=datetime.now(timezone.utc),
                seq_number=sample_idx,
                sample_rate=sfreq,
                channels=raw.ch_names,
                sample_data=sample_data,
                classification_label=None
            )
            
            # Send to Kafka
            future = producer.send('raw-eeg', eeg_sample.model_dump(mode='json'))
            future.get(timeout=5)  # Wait for confirmation
            
            # Progress indicator
            if sample_idx % int(sfreq) == 0:  # Every second
                elapsed_time = sample_idx / sfreq
                print(f"Streamed {sample_idx} samples ({elapsed_time:.1f}s of data)", end='\r')
            
            # Real-time delay
            time.sleep(sample_interval)
            
    except KeyboardInterrupt:
        print(f"\nStopped streaming at sample {sample_idx}")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        producer.flush()
        producer.close()
        print("Producer closed.")


def main():
    parser = argparse.ArgumentParser(description="Real-time EDF Sample Producer")
    parser.add_argument('--edf-file', required=True, help='Path to EDF file')
    parser.add_argument('--bootstrap-servers', default='localhost:9092', help='Kafka servers')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed factor (1.0 = real-time)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.edf_file):
        print(f"Error: EDF file not found: {args.edf_file}")
        sys.exit(1)
    
    stream_edf_realtime(args.edf_file, args.bootstrap_servers, args.speed)


if __name__ == "__main__":
    main()
