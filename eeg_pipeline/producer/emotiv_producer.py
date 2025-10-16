#!/usr/bin/env python3
"""
Emotiv-Specific EEG Producer
Connects to Emotiv headsets via LSL with proper channel mapping and configuration.
"""

import argparse
import json
import sys
import os
import time
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional, Dict

from pylsl import StreamInlet, resolve_stream, StreamInfo
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Add project root to path
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from schemas.eeg_schemas import EEGSample

# Emotiv channel configurations
EMOTIV_CONFIGS = {
    "14": [
        "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
        "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
    ],
    "32": [
        "AF3", "AF4", "F7", "F3", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6", "T7", "C3",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "Pz", "P4", "P8", "PO3",
        "PO4", "O1", "O2", "AF7", "AF8", "Fp1",
        "Fp2", "Fz"
    ]
}


def detect_emotiv_stream(timeout: float = 10.0) -> Optional[StreamInlet]:
    """
    Detect and connect to Emotiv EEG stream via LSL.
    
    Returns:
        StreamInlet if found, None otherwise
    """
    print(f"🔍 Scanning for Emotiv EEG streams (timeout: {timeout}s)...")
    
    # Try to find Emotiv-specific streams first
    emotiv_names = ["EmotivDataStream", "Emotiv", "EPOC", "Insight", "EMOTIV"]
    
    for name_pattern in emotiv_names:
        print(f"   Looking for '{name_pattern}'...")
        streams = resolve_stream('name', name_pattern, timeout=2.0)
        if streams:
            print(f"✅ Found Emotiv stream: '{streams[0].name()}'")
            return StreamInlet(streams[0], max_chunklen=1)
    
    # Fall back to EEG type
    print("   Looking for any EEG stream...")
    streams = resolve_stream('type', 'EEG', timeout=timeout)
    
    if not streams:
        print("❌ No Emotiv or EEG streams found")
        print("\n💡 Troubleshooting:")
        print("   1. Is EmotivPRO or EmotivBCI running?")
        print("   2. Is LSL streaming enabled in Emotiv software?")
        print("   3. Is the headset connected and turned on?")
        print("   4. Check headset battery level")
        return None
    
    # Check if any stream looks like Emotiv
    for stream_info in streams:
        name = stream_info.name().lower()
        if any(keyword in name for keyword in ['emotiv', 'epoc', 'insight']):
            print(f"✅ Found Emotiv-like stream: '{stream_info.name()}'")
            return StreamInlet(stream_info, max_chunklen=1)
    
    # Use first EEG stream as fallback
    print(f"⚠️  Using generic EEG stream: '{streams[0].name()}'")
    print("   (This might not be an Emotiv headset)")
    return StreamInlet(streams[0], max_chunklen=1)


def get_emotiv_channel_mapping(inlet: StreamInlet) -> Dict[str, List[str]]:
    """
    Extract and validate channel mapping from Emotiv stream.
    
    Returns:
        Dictionary with 'detected_channels', 'standard_channels', 'mapping_type'
    """
    info = inlet.info()
    channel_count = info.channel_count()
    
    # Try to get channel names from stream
    desc = info.desc()
    channels_node = desc.child("channels")
    
    detected_channels = []
    if not channels_node.empty():
        channel = channels_node.child("channel")
        for _ in range(channel_count):
            label = channel.child_value("label")
            if label:
                detected_channels.append(label)
            else:
                detected_channels.append(f"Ch{len(detected_channels)+1}")
            channel = channel.next_sibling()
    else:
        # No channel info, use generic names
        detected_channels = [f"Ch{i+1}" for i in range(channel_count)]
    
    # Determine Emotiv model
    mapping_type = "unknown"
    standard_channels = detected_channels
    
    if channel_count == 14:
        mapping_type = "emotiv-14"
        standard_channels = EMOTIV_CONFIGS["14"]
        print(f"📊 Detected: Emotiv EPOC/Insight (14 channels)")
    elif channel_count == 32:
        mapping_type = "emotiv-32"
        standard_channels = EMOTIV_CONFIGS["32"]
        print(f"📊 Detected: Emotiv Flex/Extended (32 channels)")
    else:
        print(f"⚠️  Non-standard channel count: {channel_count}")
        print(f"   Using detected names: {detected_channels[:5]}...")
    
    return {
        "detected_channels": detected_channels,
        "standard_channels": standard_channels,
        "mapping_type": mapping_type,
        "channel_count": channel_count
    }


def check_signal_quality(sample_data: List[float]) -> Dict[str, any]:
    """
    Basic signal quality check for Emotiv data.
    
    Returns:
        Dictionary with quality metrics
    """
    import numpy as np
    
    data_array = np.array(sample_data)
    
    # Check for flat signal (common issue)
    is_flat = np.std(data_array) < 0.1
    
    # Check for extreme values (saturation)
    is_saturated = np.any(np.abs(data_array) > 10000)  # μV threshold
    
    # Check for reasonable range (EEG is typically -200 to +200 μV)
    in_range = np.all(np.abs(data_array) < 500)
    
    quality = "good"
    if is_flat:
        quality = "flat"
    elif is_saturated:
        quality = "saturated"
    elif not in_range:
        quality = "out_of_range"
    
    return {
        "quality": quality,
        "std": float(np.std(data_array)),
        "min": float(np.min(data_array)),
        "max": float(np.max(data_array)),
        "mean": float(np.mean(data_array))
    }


def stream_emotiv_eeg(
    bootstrap_servers: str,
    timeout: float = 10.0,
    check_quality: bool = True,
    quality_check_interval: int = 100
):
    """
    Stream Emotiv EEG data from LSL to Kafka with proper validation.
    """
    
    # Find Emotiv stream
    inlet = detect_emotiv_stream(timeout)
    if inlet is None:
        print("\n❌ Failed to connect to Emotiv headset")
        print("\n📖 Setup Guide:")
        print("   1. Open EmotivPRO or EmotivBCI software")
        print("   2. Connect your Emotiv headset")
        print("   3. Enable LSL streaming:")
        print("      - EmotivPRO: Settings → Data Streams → LSL → Enable")
        print("      - EmotivBCI: Settings → Enable LSL")
        print("   4. Re-run this script")
        sys.exit(1)
    
    # Get stream info
    info = inlet.info()
    sfreq = info.nominal_srate()
    
    # Get and validate channel mapping
    channel_info = get_emotiv_channel_mapping(inlet)
    channels = channel_info["standard_channels"]
    
    print(f"\n📡 Stream Configuration:")
    print(f"   Sample Rate: {sfreq} Hz")
    print(f"   Channels ({len(channels)}): {', '.join(channels[:8])}{'...' if len(channels) > 8 else ''}")
    print(f"   Mapping: {channel_info['mapping_type']}")
    print(f"   Expected latency: {1000/sfreq:.1f}ms per sample")
    print("-" * 60)
    
    # Setup Kafka producer
    print(f"📤 Connecting to Kafka: {bootstrap_servers}")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        compression_type="lz4",
        batch_size=1,  # Real-time mode
        linger_ms=0,
    )
    
    # Session info
    session_id = uuid4()
    device_id = f"emotiv_{info.hostname()}_{os.getpid()}"
    
    print(f"🆔 Session ID: {session_id}")
    print(f"🆔 Device ID: {device_id}")
    print(f"\n▶️  Starting Emotiv EEG stream... (Press Ctrl+C to stop)\n")
    
    seq_number = 0
    samples_received = 0
    quality_issues = 0
    start_time = time.time()
    
    try:
        while True:
            # Pull sample from LSL
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            
            if sample is not None:
                # Check signal quality periodically
                if check_quality and samples_received % quality_check_interval == 0:
                    quality = check_signal_quality(sample)
                    if quality["quality"] != "good":
                        quality_issues += 1
                        print(f"⚠️  Signal quality issue: {quality['quality']} "
                              f"(std={quality['std']:.2f}μV)")
                        
                        if quality["quality"] == "flat":
                            print("   → Check electrode contact and impedance")
                        elif quality["quality"] == "saturated":
                            print("   → Signal saturated, check connections")
                
                # Create EEG sample message
                eeg_sample = EEGSample(
                    device_id=device_id,
                    session_id=session_id,
                    timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    seq_number=seq_number,
                    sample_rate=sfreq,
                    channels=channels,
                    sample_data=sample,
                    classification_label=None
                )
                
                # Send to Kafka
                try:
                    future = producer.send('raw-eeg', eeg_sample.model_dump(mode='json'))
                    future.get(timeout=2)
                except Exception as e:
                    print(f"❌ Kafka error: {e}")
                    continue
                
                seq_number += 1
                samples_received += 1
                
                # Progress indicator (every second)
                if samples_received % int(sfreq) == 0:
                    elapsed = time.time() - start_time
                    rate = samples_received / elapsed
                    quality_pct = 100 * (1 - quality_issues / max(1, samples_received // quality_check_interval))
                    
                    print(f"📊 {samples_received} samples | "
                          f"{elapsed:.1f}s | "
                          f"{rate:.1f} Hz | "
                          f"Quality: {quality_pct:.0f}%", end='\r')
            else:
                # Timeout - no sample received
                print("⏱️  Waiting for data...", end='\r')
    
    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopped by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Statistics
        elapsed = time.time() - start_time
        print(f"\n📈 Session Statistics:")
        print(f"   Total samples: {samples_received}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Average rate: {samples_received / elapsed:.1f} Hz")
        print(f"   Quality issues: {quality_issues}")
        
        # Cleanup
        producer.flush()
        producer.close()
        inlet.close_stream()
        print(f"✅ Emotiv stream closed")


def main():
    parser = argparse.ArgumentParser(
        description="Emotiv EEG Producer for LSL → Kafka streaming"
    )
    parser.add_argument(
        '--bootstrap-servers', default='localhost:9092',
        help='Kafka bootstrap servers (default: localhost:9092)'
    )
    parser.add_argument(
        '--timeout', type=float, default=10.0,
        help='Timeout for finding LSL streams (seconds, default: 10.0)'
    )
    parser.add_argument(
        '--no-quality-check', action='store_true',
        help='Disable signal quality checking'
    )
    parser.add_argument(
        '--quality-interval', type=int, default=100,
        help='Check quality every N samples (default: 100)'
    )
    
    args = parser.parse_args()
    
    stream_emotiv_eeg(
        bootstrap_servers=args.bootstrap_servers,
        timeout=args.timeout,
        check_quality=not args.no_quality_check,
        quality_check_interval=args.quality_interval
    )


if __name__ == "__main__":
    main()
