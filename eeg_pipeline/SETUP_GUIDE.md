# EEG Pipeline - Quick Setup Guide

## Choose Your Setup

### 1. Basic Setup (Single Computer)

**For testing with sample EDF files**

```bash
./setup_basic.sh
```

- Sets up Kafka and Python environment
- Tests with included sample data
- Good for development and testing

### 2. Two-Device Setup (Emotiv → KUKA)

**For real EEG headset → robotic arm control**

```bash
./setup_two_devices.sh
```

- Device 1: EEG data collection (Emotiv headset)
- Device 2: Data processing & KUKA arm control
- Production-ready real-time pipeline

### 3. Hardware-Only Setup

**For advanced users with existing Kafka**

```bash
./setup_realtime.sh
```

- Just installs LSL for hardware integration
- Assumes Kafka already running

### 4. Demo

**See the complete pipeline in action**

```bash
./demo.sh
```

- Shows KUKA arm responding to EEG signals
- Uses sample data (no hardware needed)
- Great for demonstrations

## Most Common Use Cases

**First time user**: `./setup_two_devices.sh`
**Testing/Development**: `./setup_basic.sh`
**Show someone how it works**: `./demo.sh`
