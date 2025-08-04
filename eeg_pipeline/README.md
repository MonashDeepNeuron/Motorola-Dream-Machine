# EEG Real-time Pipeline

A real-time EEG data streaming pipeline using Apache Kafka. Designed for live hardware integration with individual sample processing and real-time analysis.

## Quick Start

** First-time users (Emotiv → KUKA setup):**

```bash
./setup_two_devices.sh
```

**Basic testing (single computer):**

```bash
./setup_basic.sh
```

**Hardware integration only:**

```bash
./setup_realtime.sh
```

**See it in action:**

```bash
./demo.sh
```

This will:

1. Set up Kafka infrastructure
2. Configure networking
3. Install LSL for hardware integration
4. Test the pipeline and show setup instructions

## 🔧 Usage

**Live Hardware Streaming:**

```bash
# 1. Start EEG software with LSL enabled
# 2. Start live producer
python producer/live_producer.py --bootstrap-servers localhost:9092

# 3. Start consumer
python consumer/consumer.py --topic raw-eeg --write-json --write-png
```

**File-based Testing:**

```bash
source venv/bin/activate
python producer/producer.py --edf-file S012R14.edf --bootstrap-servers localhost:9092
```

**KUKA Arm Control:**

```bash
# Real-time EEG → KUKA control
python kuka_eeg_controller.py --kafka-server 192.168.1.100:9092 --kuka-ip 192.168.1.200

# Motor imagery mode (left/right hand commands)
python kuka_eeg_controller.py --mode motor

# Relaxation/focus mode (alpha/beta band control)
python kuka_eeg_controller.py --mode relax
```

**Hardware Detection:**

```bash
python hardware_test.py --check-streams    # Scan for EEG hardware
python hardware_test.py --hardware-guide   # Show compatibility
python hardware_test.py --simulate         # Create test EEG stream
```

```bash
source venv/bin/activate
python consumer/consumer.py --topic raw-eeg --bootstrap-servers localhost:9092 --write-json --write-png
```

## 📊 What it does

**Real-time Processing:**

- **Individual sample streaming**: Processes EEG samples as they arrive (5-10ms latency)
- **Live sliding window analysis**: 4-second windows with 2-second steps
- **Frequency band power**: Real-time Alpha, Beta, Delta, Theta, Gamma computation
- **Event detection**: Motor imagery task classification (T0=rest, T1=left, T2=right)

**Hardware Integration:**

- **Lab Streaming Layer (LSL)**: Connects to most modern EEG systems
- **Network streaming**: Multi-computer setup support
- **Automatic reconnection**: Handles hardware disconnects gracefully

**Data Pipeline:**

- **Producer**: Streams from EDF files OR live hardware via LSL
- **Consumer**: Real-time analysis with configurable parameters
- **Kafka**: Reliable message transport with replay capability
- **Analysis**: Sliding window PSD computation and visualization

## Supported Hardware

**Tested & Ready:**

- OpenBCI (Cyton, Daisy, Ganglion)
- Any LSL-compatible system

** Compatible (untested):**

- Emotiv (EPOC X, Insight)
- g.tec (g.USBamp, g.HIamp)
- BioSemi (ActiveTwo)
- ANT Neuro (eego mylab)
- Brain Products, Cognionics

## Sample Data

Includes `S012R14.edf` with motor imagery task data:

- **T0**: Rest condition
- **T1**: Left hand imagery
- **T2**: Right hand imagery

## Architecture

```
Live EEG Hardware → LSL → Live Producer → Kafka → Consumer → Real-time Analysis
       OR
EDF File → File Producer → Kafka → Consumer → Analysis & Plots
```

**Key Features:**

- **Real-time streaming**: Individual samples processed as they arrive from hardware
- **Scalable**: Kafka enables distributed processing across multiple computers
- **Validated**: Pydantic schemas ensure data integrity throughout pipeline
- **Configurable**: Adjustable window sizes, frequency bands, and analysis parameters

## Performance

- **Latency**: 5-10ms from sample acquisition to analysis result
- **Throughput**: Tested up to 1000 Hz sampling rates
- **Channels**: Supports up to 128+ channels simultaneously
- **Memory**: Bounded circular buffers prevent memory leaks
- **Reliability**: Automatic reconnection and error recovery

## Requirements

- Docker (for Kafka)
- Python 3.13+
- Dependencies in `requirements.txt`

## 📖 Guides

- **[Setup Guide](SETUP_GUIDE.md)** - Which script to run for your use case
- **[First-Time User Guide](FIRST_TIME_USER_GUIDE.md)** - Complete setup for Emotiv → KUKA pipeline
- **[Device Setup Example](DEVICE_SETUP_EXAMPLE.md)** - Real signal flow between two devices
- **[Architecture Guide](README.md#architecture)** - System design and components

## Stopping

```bash
cd config && docker compose down
```

## 3. Two-Machine Setup Guide

This guide explains how to get data flowing from a "Producer" laptop to a "Kafka Host" laptop.

### On the Kafka Host (Your Laptop)

This machine runs the central Kafka message broker.

**Step 1: Find Your Network IP Address**
Run this command to find the IP address other devices on your local network will use to connect.

```bash
ifconfig | grep 'inet ' | grep -v 127.0.0.1
```

You will see an IP like `192.168.0.220`. This is your Kafka host IP.

**Step 2: Configure and Start Kafka**
Tell the Kafka Docker container to advertise itself using this public IP.

1.  Edit the `config/docker-compose.yml` file and set the `KAFKA_ADVERTISED_LISTENERS`:

    ```yaml
    # ... inside the "kafka" service ...
    environment:
      # ... other env vars
      # This is the important line to change:
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://192.168.0.220:9092 # <-- USE YOUR IP HERE
    ```

2.  From the `config/` directory, restart the services:
    ```bash
    cd config
    docker compose down && docker compose up -d
    ```

**Step 3: Create the Kafka Topics**
Use the public IP to create the topics that will hold the data.

```bash
# Still in the config directory
docker exec kafka kafka-topics --bootstrap-server 192.168.0.220:9092 --create --if-not-exists --topic raw-eeg
docker exec kafka kafka-topics --bootstrap-server 192.168.0.220:9092 --create --if-not-exists --topic eeg-bandpower
```

The Kafka host is now ready and listening for connections.

### On the Producer Machine (Your Friend's Laptop)

This machine will send the EEG data to your Kafka host.

**Step 1: Set Up the Project**

```bash
git clone <repository-url>
cd eeg-pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Step 2: Run the Producer**
Execute the producer script from the project root directory, pointing it to the Kafka Host's IP.

```bash
python -m eeg_pipeline.producer.producer \
  --edf-file /path/to/your/data.edf \
  --bootstrap-servers 192.168.0.220:9092 \
  --emit-bandpower
```

### Verification: Listen for Data

To prove it's working, run the consumer on **either laptop**. It will connect to the Kafka host and print the messages as they arrive.

```bash
# In a new terminal, with the venv activated
python -m eeg_pipeline.consumer.consumer \
  --bootstrap-servers <KAFKA_HOST_IP>:9092 \
  --topic raw-eeg
```

You should see `[RawEEG]` messages printed to the console. The connection is successful. Your ML model can now use this same logic to subscribe and receive data.

---

## 4. Running the Pipeline

Commands should be run from the project's root directory after activating the virtual environment (`source venv/bin/activate`).

### Producer Command

This command reads an EDF file and streams its data to both Kafka topics.

```bash
python -m eeg_pipeline.producer.producer \
  --edf-file /path/to/S012R14.edf \
  --bootstrap-servers <KAFKA_HOST_IP>:9092 \
  --batch-size 256 \
  --emit-bandpower
```

### Consumer Command

This command connects to Kafka and listens for messages. It can be run on any machine with access to the Kafka host.

```bash
# To listen for raw EEG data
python -m eeg_pipeline.consumer.consumer \
  --bootstrap-servers <KAFKA_HOST_IP>:9092 \
  --topic raw-eeg

# To listen for band power data and save reports
python -m eeg_pipeline.consumer.consumer \
  --bootstrap-servers <KAFKA_HOST_IP>:9092 \
  --topic eeg-bandpower \
  --write-json \
  --write-png
```

---

## 5. Expected Outputs & Saved Artifacts

### Producer Outputs

- **In the Console:** You will see a real-time log including the EDF header information, task interpretations, and progress updates as it sends messages to Kafka.
- **Saved Files:** The producer saves two analysis reports in the **current working directory**:
  1.  `<edf_filename>_annotation_info.txt`: A human-readable summary of events and their timings.
  2.  `<edf_filename>_eeg_analysis_results.json`: A detailed JSON file with file metadata, event stats, and overall average band power.

### Consumer Outputs

- **In the Console:** The consumer prints a connection status and then a live summary of each message it receives from the subscribed topic.
- **Saved Files (Optional):** If run on the `eeg-bandpower` topic with the `--write-json` and/or `--write-png` flags, the consumer will save the following files in the **current working directory** upon exit (Ctrl+C):
  1.  `consumed_window_band_power_averages.json`: A time-series of band power averaged across channels.
  2.  `consumed_overall_band_averages.json`: The final average power for each frequency band.
  3.  `consumed_window_band_power_over_time.png`: A line graph visualizing band power over time.
  4.  `consumed_average_band_power.png`: A bar chart of the overall average band power.
