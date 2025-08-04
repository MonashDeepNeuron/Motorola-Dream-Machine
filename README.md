# Motorola-Dream-Machine

Dream machine project from Motorola grant, using an EEG headset to process brain signals in order to function robotic parts such as an arm. This project is an exploration of real-time brain-computer interfaces.

## EEG Pipeline Architecture

The core of this project is a real-time, streaming pipeline that uses Lab Streaming Layer (LSL) and Kafka to process live EEG data.

### Components

1.  **LSL Data Source (External)**:

    - This is not part of the repository but is a prerequisite. It can be:
      - **Real EEG Hardware**: An EEG amplifier with acquisition software (e.g., OpenBCI GUI, BrainVision Recorder) that broadcasts its data onto the local network via an LSL stream.
      - **Simulated Stream**: A tool like OpenViBE can be used to stream an existing `.edf` file as a live LSL stream for development and testing.

2.  **Producer (`eeg_pipeline/producer/producer.py`)**:

    - **Architectural Role**: Acts as a bridge between the LSL network and the Kafka messaging system.
    - It continuously searches the local network for an LSL stream of type 'EEG'.
    - Once a stream is found, it connects, pulls data chunks in real-time, and forwards each individual `EEGSample` to a Kafka topic.
    - It no longer reads from files; it is a pure real-time listener.

3.  **Consumer (`eeg_pipeline/consumer/consumer.py`)**:

    - **Architectural Role**: The real-time analysis engine.
    - It subscribes to the Kafka topic (`raw-eeg` by default) and consumes the `EEGSample` stream.
    - It dynamically windows the incoming data and calculates the Power Spectral Density (PSD) for each window using Welch's method.
    - Upon termination (Ctrl-C), it saves summary analysis artifacts (JSON data and PNG plots) to the root directory.

4.  **Kafka**:
    - The robust, high-throughput message bus that decouples the LSL producer from the analysis consumer.
    - A simple `docker-compose.yml` is provided in `eeg_pipeline/config/` to run Kafka.

---

## How to Run the Real-Time Pipeline

The workflow now mirrors a typical real-time BCI experiment.

### 1. Start an LSL Stream Source

This is the most critical step. The producer is a listener, so something must be broadcasting data. You have two options:

- **With Real Hardware**: Start your EEG amplifier and its acquisition software. Find the setting to enable LSL streaming.
- **For Development**: Use a tool like **OpenViBE** to create a simple scenario that reads an `.edf` file and streams it to the network using an LSL output block.

### 2. Start Kafka Services

This is the same as before. Make sure Docker is running.

```bash
cd eeg_pipeline/config
docker-compose up -d
```

### 3. Run the Real-Time Producer

Activate your Python virtual environment. The producer no longer takes a file path; it automatically discovers the LSL stream.

```bash
# Activate your environment
source eeg_pipeline/venv/bin/activate

# Run the producer
python eeg_pipeline/producer/producer.py --bootstrap-servers localhost:9092
```

The script will print "Looking for an EEG stream via LSL..." and wait. Once your LSL source is broadcasting, it will connect and begin forwarding data to Kafka.

### 4. Run the Consumer

In a separate terminal, activate the environment and run the consumer. It will process the live data from Kafka.

```bash
# Activate your environment
source eeg_pipeline/venv/bin/activate

# Run the consumer
python eeg_pipeline/consumer/consumer.py --topic raw-eeg --bootstrap-servers localhost:9092 --write-json --write-png
```

The consumer will now perform continuous analysis on the live data. When you are finished, stop the producer and consumer with `Ctrl-C`. The consumer will save its final analysis files upon termination.
