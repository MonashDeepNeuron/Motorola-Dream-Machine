#!/usr/bin/env python3
"""
EEG Hardware Integration Guide and Test Suite
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

def check_lsl_streams():
    """Check for available LSL streams on the network."""
    try:
        from pylsl import resolve_streams
        print("🔍 Scanning for LSL streams...")
        
        streams = resolve_streams(wait_time=3.0)
        
        if not streams:
            print("❌ No LSL streams found on network")
            print("\nTo test with simulated data:")
            print("1. Install LSL examples: pip install pylsl")
            print("2. Run LSL example app or use OpenBCI GUI")
            return False
        
        print(f"✅ Found {len(streams)} LSL stream(s):")
        for i, stream in enumerate(streams):
            print(f"  {i+1}. Name: '{stream.name()}'")
            print(f"     Type: {stream.type()}")
            print(f"     Channels: {stream.channel_count()}")
            print(f"     Sample Rate: {stream.nominal_srate()} Hz")
            print(f"     Host: {stream.hostname()}")
            print()
        
        return True
        
    except ImportError:
        print("❌ pylsl not installed")
        return False
    except Exception as e:
        print(f"❌ Error scanning LSL: {e}")
        return False


def test_hardware_compatibility():
    """Test compatibility with common EEG hardware."""
    
    print("🧠 EEG Hardware Compatibility Guide")
    print("=" * 50)

    compatible_systems = {
        "OpenBCI": {
            "models": ["Cyton (8-ch)", "Daisy (16-ch)", "Ganglion (4-ch)"],
            "connection": "OpenBCI GUI with LSL plugin",
            "setup": "1. Install OpenBCI GUI\n2. Enable LSL streaming\n3. Start our live_producer.py",
            "tested": True
        },
        "Emotiv": {
            "models": ["EPOC X", "Insight", "EPOC+"],
            "connection": "EmotivPRO software with LSL",
            "setup": "1. Install EmotivPRO\n2. Enable LSL streaming\n3. Start our live_producer.py",
            "tested": False
        },
        "NeuroSky": {
            "models": ["MindWave", "MindWave Mobile"],
            "connection": "Third-party LSL bridge",
            "setup": "1. Install NeuroSky SDK\n2. Use community LSL bridge\n3. Start our live_producer.py",
            "tested": False
        },
        "g.tec": {
            "models": ["g.USBamp", "g.HIamp"],
            "connection": "Simulink with LSL or direct SDK",
            "setup": "1. Use g.tec Simulink drivers\n2. Configure LSL output\n3. Start our live_producer.py",
            "tested": False
        },
        "BioSemi": {
            "models": ["ActiveTwo"],
            "connection": "ActiView software with LSL",
            "setup": "1. Install ActiView\n2. Configure LSL streaming\n3. Start our live_producer.py",
            "tested": False
        },
        "ANT Neuro": {
            "models": ["eego mylab", "NOVA"],
            "connection": "eegosoftware with LSL",
            "setup": "1. Use eego software\n2. Enable LSL output\n3. Start our live_producer.py",
            "tested": False
        }
    }
    
    for system, info in compatible_systems.items():
        status = "✅ TESTED" if info["tested"] else "⚪ UNTESTED"
        print(f"\n{system} {status}")
        print(f"  Models: {', '.join(info['models'])}")
        print(f"  Connection: {info['connection']}")
        print(f"  Setup Steps:")
        for step in info['setup'].split('\n'):
            print(f"    {step}")
    
    print("\n" + "=" * 50)
    print("💡 Key Requirements:")
    print("  - EEG hardware with LSL support (most modern systems)")
    print("  - Lab Streaming Layer (LSL) - already installed")
    print("  - Network connection between EEG computer and our pipeline")
    print("\n🔧 For DIY/Custom hardware:")
    print("  - Implement LSL outlet in your acquisition software")
    print("  - Or modify our producer to use your hardware's SDK directly")


def simulate_eeg_stream():
    """Create a simulated EEG stream for testing."""
    try:
        from pylsl import StreamInfo, StreamOutlet
        import numpy as np
        import threading
        
        print("🎭 Starting simulated EEG stream...")
        
        # Create stream info
        info = StreamInfo(
            name="SimulatedEEG",
            type="EEG", 
            channel_count=8,
            nominal_srate=250,
            channel_format='float32',
            source_id="sim_eeg_001"
        )
        
        # Add channel names
        desc = info.desc()
        channels = desc.append_child("channels")
        for ch in ["Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2"]:
            channels.append_child("channel").append_child_value("label", ch)
        
        # Create outlet
        outlet = StreamOutlet(info)
        
        def stream_data():
            srate = 250
            t = 0
            print(f"✅ Simulated EEG streaming at {srate} Hz with 8 channels")
            print("   Stop with Ctrl-C")
            
            try:
                while True:
                    # Generate simulated EEG-like data
                    # Mix of alpha (10 Hz), theta (6 Hz), and noise
                    alpha = np.sin(2 * np.pi * 10 * t) * 10  # 10 Hz alpha
                    theta = np.sin(2 * np.pi * 6 * t) * 5    # 6 Hz theta  
                    noise = np.random.normal(0, 2, 8)        # Random noise
                    
                    # Create 8-channel sample
                    sample = [alpha + theta + noise[i] for i in range(8)]
                    
                    # Send sample
                    outlet.push_sample(sample)
                    
                    # Update time
                    t += 1.0 / srate
                    time.sleep(1.0 / srate)
                    
            except KeyboardInterrupt:
                print("\n🛑 Simulated stream stopped")
        
        # Start streaming in background
        thread = threading.Thread(target=stream_data, daemon=True)
        thread.start()
        
        return True
        
    except ImportError:
        print("❌ Cannot create simulated stream: pylsl not available")
        return False
    except Exception as e:
        print(f"❌ Error creating simulated stream: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="EEG Hardware Integration Tools")
    parser.add_argument('--check-streams', action='store_true', 
                       help='Scan for available LSL streams')
    parser.add_argument('--hardware-guide', action='store_true',
                       help='Show hardware compatibility guide')
    parser.add_argument('--simulate', action='store_true',
                       help='Start simulated EEG stream for testing')
    parser.add_argument('--test-pipeline', action='store_true',
                       help='Run complete pipeline test with simulated data')
    
    args = parser.parse_args()
    
    if args.check_streams:
        check_lsl_streams()
    elif args.hardware_guide:
        test_hardware_compatibility()
    elif args.simulate:
        if simulate_eeg_stream():
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Simulation stopped")
    elif args.test_pipeline:
        print("🧪 Full Pipeline Test")
        print("1. Starting simulated EEG stream...")
        if simulate_eeg_stream():
            time.sleep(2)
            print("2. Starting live producer...")
            # Here you would start the live producer
            print("3. Run: python producer/live_producer.py")
            print("4. Run: python consumer/consumer.py --topic raw-eeg")
    else:
        print("EEG Hardware Integration Tools")
        print("Usage:")
        print("  --check-streams    : Scan for LSL streams")
        print("  --hardware-guide   : Show compatible hardware")
        print("  --simulate         : Start test EEG stream")
        print("  --test-pipeline    : Full pipeline test")


if __name__ == "__main__":
    main()
