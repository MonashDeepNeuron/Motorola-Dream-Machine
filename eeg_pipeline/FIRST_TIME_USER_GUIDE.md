# First-Time User Guide: Emotiv EEG → KUKA Arm Setup

## 🎯 Overview

This guide will help you set up a two-device system where:

- **Device 1**: Collects EEG data from an Emotiv headset
- **Device 2**: Processes the data and controls a KUKA robotic arm

**Expected Setup Time**: 15-20 minutes
**Skill Level**: Beginner-friendly with copy-paste commands

## What You'll Need

### Hardware

- Emotiv EEG headset
- 2 computers connected to the same network
- KUKA robotic arm with network connection

### Software Prerequisites

- Python 3.13+ on both devices
- Docker on Device 2 (Kafka host)
- Emotiv software (EmotivBCI or EMOTIV Launcher)

## 🚀 Quick Setup (5 Minutes)

### Device 1: Emotiv Data Collection

1. **Download this project:**

```bash
cd eeg_pipeline
```

2. **Set up Python environment:**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Install LSL library:**

```bash
# macOS
brew install labstreaminglayer/tap/lsl

# Linux
sudo apt-get install liblsl-dev

# Windows
# Download LSL from: https://github.com/sccn/liblsl/releases
```

### Device 2: Data Processing & KUKA Control

1. **Set up the same project:**

```bash
cd eeg_pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Start Kafka (message broker):**

```bash
./setup_realtime.sh
```

This will:

- Start Docker containers for Kafka
- Create necessary topics
- Show your network IP address
- Test the connection

## 🔧 Step-by-Step Configuration

### Step 1: Configure Device 2 (Kafka Host)

**Find your network IP:**

```bash
ifconfig | grep 'inet ' | grep -v 127.0.0.1
```

You'll see something like: `192.168.1.100` - **write this down!**

**Edit Kafka configuration:**

```bash
# Edit config/docker-compose.yml
# Replace the KAFKA_ADVERTISED_LISTENERS line with your IP:
KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://192.168.1.100:9092
```

**Restart Kafka with your IP:**

```bash
cd config
docker compose down && docker compose up -d
```

### Step 2: Set Up Emotiv on Device 1

**Start Emotiv software:**

1. Open EmotivBCI or EMOTIV Launcher
2. Connect your headset
3. **Enable LSL streaming**:
   - Look for "LSL" or "Lab Streaming Layer" in settings
   - Enable EEG data streaming
   - Note the stream name (usually "EmotivDataStream-EEG")

**Test the connection:**

```bash
python hardware_test.py --check-streams
```

You should see: `Found EEG stream: EmotivDataStream-EEG`

### Step 3: Start Data Flow

**On Device 1 (Emotiv):**

```bash
python producer/live_producer.py --bootstrap-servers 192.168.1.100:9092
```

**On Device 2 (Processing):**

```bash
python consumer/consumer.py --topic raw-eeg --bootstrap-servers localhost:9092 --write-json
```

You should see real-time EEG data flowing!

## 🤖 KUKA Arm Integration

### Option 1: Real-time Band Power Control

Create a KUKA controller that responds to EEG states:

```python
# kuka_controller.py
import json
import time
from kafka import KafkaConsumer
from kuka_robotics import KukaArm  # Your KUKA library

class EEGKukaController:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'eeg-bandpower',
            bootstrap_servers=['192.168.1.100:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.kuka = KukaArm("192.168.1.200")  # KUKA IP address

    def run(self):
        for message in self.consumer:
            band_power = message.value

            # Extract alpha power (relaxation indicator)
            alpha_power = band_power['Alpha']

            # Control KUKA based on alpha levels
            if alpha_power > 0.8:  # High relaxation
                self.kuka.move_to_position("rest")
            elif alpha_power > 0.5:  # Medium
                self.kuka.move_to_position("ready")
            else:  # Low relaxation (focused)
                self.kuka.move_to_position("active")

# Run the controller
controller = EEGKukaController()
controller.run()
```

### Option 2: Motor Imagery Control

Use the built-in motor imagery detection:

```python
# motor_imagery_kuka.py
from kafka import KafkaConsumer
import json

def control_kuka_from_imagery():
    consumer = KafkaConsumer(
        'raw-eeg',
        bootstrap_servers=['192.168.1.100:9092'],
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    for message in consumer:
        eeg_sample = message.value

        # Check for motor imagery events
        if 'event_annotation' in eeg_sample:
            event = eeg_sample['event_annotation']

            if event == 'T1':  # Left hand imagery
                kuka.move_left()
            elif event == 'T2':  # Right hand imagery
                kuka.move_right()
            elif event == 'T0':  # Rest
                kuka.stop()
```

## 📊 What You'll See

### Terminal Output Examples

**Device 1 (Emotiv Producer):**

```
✅ Connected to Kafka at 192.168.1.100:9092
🧠 Found EEG stream: EmotivDataStream-EEG
📡 Streaming 14 channels at 128 Hz
⚡ Sent sample 1234 (Alpha: 0.67, Beta: 0.43)
```

**Device 2 (Consumer):**

```
📥 Connected to raw-eeg topic
🧠 Processing EEG sample from timestamp 1691234567.123
📊 Window 45: Alpha=0.67 Beta=0.43 Delta=0.23 Theta=0.34 Gamma=0.12
💾 Saved analysis to eeg_analysis_results.json
```

### Real-time Performance

- **Data Rate**: 128 samples/second from Emotiv
- **Latency**: 5-10ms from headset to KUKA command
- **Reliability**: Automatic reconnection if devices disconnect

## 🔍 Troubleshooting

### Common Issues

**"No EEG streams found"**

- Check Emotiv software is running
- Verify LSL is enabled in Emotiv settings
- Try: `python hardware_test.py --simulate` to test pipeline

**"Connection refused to Kafka"**

- Verify IP address in commands
- Check both devices are on same network
- Restart Kafka: `cd config && docker compose restart`

**"KUKA not responding"**

- Check KUKA network connection
- Verify KUKA IP address
- Test KUKA connection independently

### Getting Help

**Test everything step by step:**

```bash
# Test 1: Check Emotiv connection
python hardware_test.py --check-streams

# Test 2: Test without hardware
python hardware_test.py --simulate

# Test 3: Check Kafka connection
python producer/producer.py --edf-file S012R14.edf --bootstrap-servers 192.168.1.100:9092

# Test 4: Verify consumer works
python consumer/consumer.py --topic raw-eeg --bootstrap-servers 192.168.1.100:9092
```

## 🎉 Success Criteria

You'll know it's working when:

1. ✅ Emotiv headset streams data to Device 1
2. ✅ Device 1 shows "Streaming X channels at Y Hz"
3. ✅ Device 2 shows "Processing EEG sample from timestamp..."
4. ✅ KUKA arm responds to EEG state changes
5. ✅ JSON files are saved with real-time analysis results

## 🚀 Next Steps

Once working:

- Calibrate EEG thresholds for your specific user
- Add safety limits to KUKA movements
- Create custom analysis for your specific application

**Total setup time: 15-20 minutes for complete EEG → KUKA pipeline!**
