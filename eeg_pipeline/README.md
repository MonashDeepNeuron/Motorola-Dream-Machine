# EEG Data Processing Pipeline with Kafka

This project provides a robust, lightweight, end-to-end Kafka-based pipeline for ingesting, validating, processing, and relaying EEG data from EDF files. It handles raw EEG signals and can also compute and stream windowed band power information.

## Core Features

- **EDF Ingestion**: Reads EDF and EDF+ files, including associated `.event` files for annotations (via MNE-Python).
- **Producer (`producer.producer`)**:
  - Parses and displays detailed EDF header information.
  - Processes EDF annotations (e.g., 'T0', 'T1', 'T2') to generate per-sample classification labels, interpreting task-specific event types based on EDF filename conventions (run number).
  - Publishes raw EEG data (in Volts) as `EEGBatch` messages, including per-sample classification labels, to a Kafka topic (default: `raw-eeg`).
  - Optionally computes Power Spectral Density (PSD, in $\mu V^2/Hz$) for standard frequency bands (Delta, Theta, Alpha, Beta, Gamma) using a sliding window approach. Publishes these as `WindowBandPower` messages (one per channel, per band, per window) to a separate Kafka topic (default: `eeg-bandpower`).
  - Saves local analysis reports after processing an EDF file:
    - `[edf_basename]_annotation_info.txt`: Human-readable summary of annotations, event interpretations, and timings.
    - `[edf_basename]_eeg_analysis_results.json`: JSON with file information, annotation details, event-locked band power statistics (mean power in $\mu V^2$), and overall average band PSD (if bandpower emission was enabled).
- **Consumer (`consumer.consumer`)**:
  - Subscribes to either the `raw-eeg` or `eeg-bandpower` topic.
  - Validates incoming JSON messages against Pydantic schemas (`EEGBatch`, `WindowBandPower`).
  - For `EEGBatch` messages: Prints a summary of the received batch.
  - For `WindowBandPower` messages: Aggregates per-channel power values to calculate an average power across all channels for each window and band.
  - If consuming `eeg-bandpower` (with flags set), saves on exit (Ctrl+C):
    - `consumed_window_band_power_averages.json`: Window-by-window average PSD per band.
    - `consumed_overall_band_averages.json`: Overall average PSD per band.
    - `consumed_average_band_power.png`: Bar plot of overall average band power.
    - `consumed_window_band_power_over_time.png`: Line plot of band power over time.
- **Data Integrity**: Uses Pydantic for schema definition and validation of Kafka messages.
- **Modular Analysis**: Core signal processing logic (header parsing, band power, event statistics) is in the `analysis` module.

---

## Prerequisites & Setup

### 1. System Requirements

- Python 3.8+
- Docker and Docker Compose

### 2. Install Python Dependencies

```bash
pip install kafka-python mne pydantic numpy scipy matplotlib
```

### 3. Start Kafka Infrastructure (Docker)

Navigate to the `config/` directory and run:

```bash
cd config
docker-compose up -d
```

This starts Zookeeper (port `2181`), Kafka (port `9092`), and Schema Registry (port `8081`).
Verify with `docker ps`.

### 4. Create Kafka Topics (First Time Only)

```bash
# Topic for raw EEG data
docker exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic raw-eeg --partitions 3 --replication-factor 1

# Topic for band power data
docker exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic eeg-bandpower --partitions 3 --replication-factor 1

docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```

---

## Running the Pipeline

Commands should be run from the project's root directory.

### 1. Producer (`producer.producer`)

**Key Arguments:**

- `--edf-file <PATH>`: **(Required)** Path to EDF.
- `--bootstrap-servers <HOST:PORT>`: Kafka brokers (default: `localhost:9092`).
- `--topic <NAME>`: Topic for `EEGBatch` (default: `raw-eeg`).
- `--batch-size <SIZE>`: Samples per `EEGBatch` (default: `256`).
- `--emit-bandpower`: Flag to enable `WindowBandPower` messages.
- `--bandpower-topic <NAME>`: Topic for `WindowBandPower` (default: `eeg-bandpower`).
- `--window-size <SEC>`: Window for PSD (default: `4.0`).
- `--step-size <SEC>`: Step for PSD windows (default: `2.0`).

**Example A: Raw EEG Only**

```bash
python -m eeg_pipeline.producer.producer \
  --edf-file path/to/your/data.edf \
  --topic raw-eeg
```

- **Outputs**: Console logs; `[edf_basename]_annotation_info.txt`, `[edf_basename]_eeg_analysis_results.json`.

**Example B: Raw EEG + Band Power**

```bash
python -m eeg_pipeline.producer.producer \
  --edf-file path/to/your/data.edf \
  --emit-bandpower
```

- **Outputs**: Console logs; `[edf_basename]_annotation_info.txt`, `[edf_basename]_eeg_analysis_results.json` (now including producer's PSD averages).

---

### 2. Consumer (`consumer.consumer`)

**Key Arguments:**

- `--bootstrap-servers <HOST:PORT>`: Kafka brokers (default: `localhost:9092`).
- `--topic <NAME>`: **(Required)** Topic to consume (e.g., `raw-eeg`, `eeg-bandpower`).
- `--group-id <ID>`: Consumer group ID (default: `eeg-consumer-group`).
- `--write-json`: For `eeg-bandpower` topic, save aggregated JSON on exit.
- `--write-png`: For `eeg-bandpower` topic, save plots on exit.
- `--window-size <SEC>` / `--step-size <SEC>`: Used for metadata in consumer's JSON outputs if writing bandpower results.

**Example C: Consuming Raw EEG**
(Best to start consumer before/during producer activity for live view)

```bash
python -m eeg_pipeline.consumer.consumer \
  --topic raw-eeg \
  --group-id raw-data-viewer
```

- **Outputs**: Console logs of `[RawEEG]` batches. Stop with `Ctrl+C`.

**Example D: Consuming Band Power & Generating Reports**
(Ensure producer ran with `--emit-bandpower`)

```bash
python -m eeg_pipeline.consumer.consumer \
  --topic eeg-bandpower \
  --group-id bandpower-results-processor \
  --write-json \
  --write-png
```

- **Outputs**: Console logs of aggregation. On `Ctrl+C`, saves `consumed_*.json` files and `consumed_*.png` plots.

---

## Kafka Message Schemas

Defined in `schemas/eeg_schemas.py` using Pydantic.

### `EEGBatch` (Topic: `raw-eeg`)

```json
{
  "device_id": "producer",
  "session_id": "uuid-string",
  "timestamp": "utc-datetime-string",
  "seq_number": 0,
  "sample_rate": 160.0,
  "channels": ["FP1", ...],
  "data": [ [-0.001, ...], ... ], // List of samples (Volts)
  "classification_labels": ["T0", ...] // Optional
}
```

### `WindowBandPower` (Topic: `eeg-bandpower`)

(One message per channel, per band, per window)

```json
{
  "device_id": "producer",
  "session_id": "uuid-string",
  "window_index": 0,
  "start_time_sec": 0.0,
  "end_time_sec": 4.0,
  "band": "delta",
  "channel_labels": ["FP1"],
  "power": [750.123] // PSD (μV²/Hz)
}
```

---

## Output File Details

### Producer Local Reports

- **`[edf_basename]_annotation_info.txt`**: Details annotations, event interpretations, timings.
- **`[edf_basename]_eeg_analysis_results.json`**:
  - `file_info`: Metadata about the EDF file.
  - `annotations_summary`: List of annotation objects.
  - `event_band_statistics_uv_sq`: Mean power ($\mu V^2$) per band per event.
  - `producer_overall_avg_band_psd_uv_sq_hz`: Average PSD ($\mu V^2/Hz$) if bandpower was emitted.

### Consumer Local Reports (from `eeg-bandpower` topic)

- **`consumed_window_band_power_averages.json`**: Window-by-window PSD, averaged across channels.
- **`consumed_overall_band_averages.json`**: Overall average PSD per band.
- **`consumed_average_band_power.png`**: Bar chart of overall average band power.
- **`consumed_window_band_power_over_time.png`**: Line plot of band power over time.

---

## Common Issues

- **Kafka Connection**: Ensure Docker containers are running and `bootstrap-servers` is correct.
- **EDF Read Errors**: Verify `--edf-file` path and file integrity.
- **Missing Annotations**: Ensure `.edf.event` file is alongside the `.edf` file.
- **Consumer: No Messages**: Confirm producer sent to the correct topic. Use a new `--group-id` to reset offsets if needed. Ensure consumer is running when messages are expected.
- **Consumer: Output Files Not Created**: Use `--write-json` / `--write-png` flags for `eeg-bandpower` consumer.
