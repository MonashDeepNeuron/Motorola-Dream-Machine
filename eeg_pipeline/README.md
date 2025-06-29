# EEG Data Pipeline for Brain-Computer Interfaces

This project provides a robust, real-time data relay pipeline for EEG signals, designed to serve as the data backbone for a brain-computer interface (BCI) controlling a robotic arm. It uses Apache Kafka to reliably stream neural data from a source computer (connected to an EEG headset) to a destination computer (running a machine learning model).

## Table of Contents

1.  [Core Concepts](#1-core-concepts)
2.  [The Data Schemas](#2-the-data-schemas-our-data-contract)
3.  [Two-Machine Setup Guide](#3-two-machine-setup-guide)
    - [On the Kafka Host Machine](#on-the-kafka-host-your-laptop)
    - [On the Producer Machine](#on-the-producer-machine-your-friends-laptop)
4.  [Running the Pipeline](#4-running-the-pipeline)
    - [Producer Command](#producer-command)
    - [Consumer Command](#consumer-command)
5.  [Expected Outputs & Saved Artifacts](#5-expected-outputs--saved-artifacts)
    - [Producer Outputs](#producer-outputs)
    - [Consumer Outputs](#consumer-outputs)

---

## 1. Core Concepts

This pipeline is built on the principle of **decoupling** using a producer-consumer architecture.

- **Apache Kafka:** Acts as the central nervous system of the pipeline. It is a distributed, persistent log that allows different components to communicate without direct knowledge of each other. This provides a resilient buffer, handles speed mismatches, and allows for data replayability.

- **Producer (`producer/producer.py`):** This application's job is to read data from a source (e.g., an EDF file or a live headset stream), process it, and serialize it into structured JSON messages that are published to Kafka.

- **Consumer (`consumer/consumer.py`):** This is a reference application that shows how any downstream service (like your ML model) would connect to Kafka, subscribe to a data stream, and deserialize the messages to use them.

---

## 2. The Data Schemas: Our Data Contract

To ensure data integrity, all messages are validated against schemas defined in `schemas/eeg_schemas.py` using Pydantic. Your ML model will consume one of these two message types.

### a. `EEGBatch` (The Raw Signal)

- **Topic:** `raw-eeg`
- **Content:** A batch of raw, unprocessed EEG sensor readings (in Volts). This is the high-fidelity digital recording of the brainwaves.
- **Use Case:** Ideal for training deep learning models (CNNs, LSTMs) that learn directly from time-series data or for archiving the original session.

### b. `WindowBandPower` (The Pre-Processed Features)

- **Topic:** `eeg-bandpower`
- **Content:** The calculated **power** within standard neurological frequency bands (Delta, Theta, Alpha, etc.). This is computed on the fly by analyzing a sliding time window of the raw signal.
- **Use Case:** Perfect for ML models that expect frequency-based features, which are often more stable and less noisy. Also great for real-time state monitoring.

---

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
