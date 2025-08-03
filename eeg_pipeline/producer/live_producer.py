#!/usr/bin/env python3
"""
Real-time LSL EEG Producer
Connects to live EEG hardware via Lab Streaming Layer (LSL)
"""

import argparse
import json
import sys
import os
import time
from uuid import uuid4
from datetime import datetime, timezone

from pylsl import StreamInlet, resolve_stream
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from schemas.eeg_schemas import EEGSample


def find_eeg_stream(stream_type='EEG', timeout=10.0):
    """Find and connect to an EEG stream on the network using LSL."""
    print(f"Looking for {stream_type} stream on LSL network...")
    print(f"Timeout: {timeout} seconds")
    
    # Resolve streams of the specified type
    streams = resolve_stream('type', stream_type, timeout=timeout)
    
    if not streams:
        available_streams = resolve_stream(timeout=2.0)  # Quick scan for any streams
        if available_streams:
            print("\nAvailable streams found:")
            for i, stream in enumerate(available_streams):
                print(f"  {i+1}. {stream.name()} (Type: {stream.type()}) "
                      f"@ {stream.hostname()} - {stream.channel_count()} channels, {stream.nominal_srate()} Hz")
            raise RuntimeError(f"No {stream_type} stream found, but other streams are available. "
                             f"Check your EEG software is streaming as type '{stream_type}'")
        else:
            raise RuntimeError(f"No LSL streams found on network. "
                             f"Make sure your EEG acquisition software is running and streaming via LSL.")
    
    # Use the first EEG stream found
    stream_info = streams[0]
    print(f"Found EEG stream: '{stream_info.name()}' @ {stream_info.hostname()}")
    print(f"  Channels: {stream_info.channel_count()}")
    print(f"  Sample Rate: {stream_info.nominal_srate()} Hz")
    print(f"  Data Type: {stream_info.channel_format()}")
    
    # Create inlet
    inlet = StreamInlet(stream_info, max_chunklen=1)  # Process one sample at a time
    return inlet


def get_channel_names(inlet):
    """Extract channel names from LSL stream info."""
    info = inlet.info()
    channel_count = info.channel_count()
    
    # Try to get channel names from stream description
    desc = info.desc()
    channels_node = desc.child("channels")
    
    channel_names = []
    if channels_node.empty():
        # No channel info available, use generic names
        channel_names = [f"Ch{i+1}" for i in range(channel_count)]
        print(f"Warning: No channel names found, using generic names: {channel_names}")
    else:
        # Extract channel names from XML description
        channel = channels_node.child("channel")
        for i in range(channel_count):
            label = channel.child_value("label")
            if label:
                channel_names.append(label)
            else:
                channel_names.append(f"Ch{i+1}")
            channel = channel.next_sibling()
    
    return channel_names


def stream_live_eeg(bootstrap_servers: str, stream_type: str = 'EEG', timeout: float = 10.0):
    """Stream live EEG data from LSL to Kafka."""
    
    try:
        # Find and connect to EEG stream
        inlet = find_eeg_stream(stream_type, timeout)
        
        # Get stream metadata
        info = inlet.info()
        sfreq = info.nominal_srate()
        channel_names = get_channel_names(inlet)
        
        print(f"\n--- Live Streaming Configuration ---")
        print(f"Sample Rate: {sfreq} Hz")
        print(f"Channels ({len(channel_names)}): {channel_names}")
        print(f"Expected interval: {1000/sfreq:.1f}ms between samples")
        print("-" * 40)
        
        # Setup Kafka producer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            compression_type="lz4",
            batch_size=1,  # Send immediately for real-time
            linger_ms=0,   # No batching delay
        )
        
        # Generate session info
        session_id = uuid4()
        device_id = f"live_eeg_{info.hostname()}_{os.getpid()}"
        
        print(f"Session ID: {session_id}")
        print(f"Device ID: {device_id}")
        print("Starting live EEG stream... (Ctrl-C to stop)")
        
        seq_number = 0
        samples_received = 0
        start_time = time.time()
        
        while True:
            # Pull sample from LSL stream
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample is not None:
                # Create EEG sample message
                eeg_sample = EEGSample(
                    device_id=device_id,
                    session_id=session_id,
                    timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    seq_number=seq_number,
                    sample_rate=sfreq,
                    channels=channel_names,
                    sample_data=sample,  # LSL provides sample as list of floats
                    classification_label=None  # Could be enhanced with marker stream
                )
                
                # Send to Kafka
                future = producer.send('raw-eeg', eeg_sample.model_dump(mode='json'))
                future.get(timeout=2)  # Wait for confirmation
                
                seq_number += 1
                samples_received += 1
                
                # Progress indicator
                if samples_received % int(sfreq) == 0:  # Every second
                    elapsed = time.time() - start_time
                    rate = samples_received / elapsed
                    print(f"Streamed {samples_received} samples in {elapsed:.1f}s "
                          f"(Rate: {rate:.1f} Hz, Expected: {sfreq} Hz)", end='\r')
            
            else:
                # No sample received (timeout)
                print(".", end="", flush=True)
                
    except KeyboardInterrupt:
        print(f"\nStopped streaming. Total samples: {samples_received}")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        try:
            producer.flush()
            producer.close()
            inlet.close_stream()
        except:
            pass
        print("Live stream closed.")


def main():
    parser = argparse.ArgumentParser(description="Live EEG Producer via LSL")
    parser.add_argument('--bootstrap-servers', default='localhost:9092', 
                       help='Kafka bootstrap servers')
    parser.add_argument('--stream-type', default='EEG', 
                       help='LSL stream type to connect to')
    parser.add_argument('--timeout', type=float, default=10.0,
                       help='Timeout for finding LSL streams (seconds)')
    
    args = parser.parse_args()
    
    stream_live_eeg(args.bootstrap_servers, args.stream_type, args.timeout)


if __name__ == "__main__":
    main()
