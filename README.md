# EEG Kafka Producer Pipeline

This project provides a pipeline to convert EEG data from `.edf` files into structured messages, compute brainwave bandpower features (delta, theta, alpha, beta, gamma), and publish this data to Apache Kafka topics. It also generates annotated reports and JSON output files for downstream processing.

---

## 📁 Project Structure

root/
├── eeg_pipeline/
│   ├── analysis/
        ├── __init.py__
        ├── plotting.py
│   │   ├── bands.py         # Defines DefaultBands, includes band power computation logic
│   │   ├── header.py        # Dumps EDF header info
│   │   └── events.py        # Event-band statistical analysis
    ├── config/
        ├── docker-compose.yml
│   └── consumer/
│       └── consumer.py   # Pydantic models: EEGBatch, WindowBandPower
├── producer/
    ├── input/
        ├── S001
            ├── S001R01.edf
            ├── S001R02.edf
            ...
        ├── S002
            ├── S002R01.edf
            ├── S002R01.edf
            ...
        ...
    ├── output/
        ├── S001R01.json
        ├── S001R02.json
        ├── ...
    ├── __init.py__
│   ├── producer.py          # Main EDF file Kafka producer
│   └── batch_producer.py    # Processes a folder of EDF files

# Purpouse

This pipeline:

    1. Ingests EEG .edf files
    2. Labels samples using EDF annotations (T0, T1, T2)
    3. Computes band powers for: Delta, Theta, Alpha, Beta, Gamma
    4. Sends structured messages to Kafka topics
    5. Generates local .json and .txt reports per file

# Frequency Bands

These have been defined in DefaultBands:

| Band  | Frequency Range (Hz) |
| ----- | -------------------- |
| Delta | 0.5 – 4              |
| Theta | 4 – 8                |
| Alpha | 8 – 13               |
| Beta  | 13 – 30              |
| Gamma | 30 – 100             |

# producer.py: EEG Gile Kafka Producer

Features:

    - Reads .edf EEG file

    - Applies per-sample movement labels using annotations

    - Optionally computes band power per window

    - Sends messages to Kafka:

        - raw-eeg (EEGBatch)

        - eeg-bandpower (WindowBandPower)

    - Outputs:

        - [filename]_annotation_info.txt

        - [filename]_eeg_analysis_results.json

# Usage

The file will need to be used through this retrospect:

python producer.py \
  --edf-file data/S001R03.edf \
  --bootstrap-servers localhost:9092 \
  --topic raw-eeg \
  --batch-size 256 \
  --emit-bandpower \
  --bandpower-topic eeg-bandpower \
  --window-size 4.0 \
  --step-size 2.0

# batch_producer.py: Batch File Processor

Features:

    1. Recursively finds .edf files
    2. Sorts by subject/run number
    3. Executes producer.py on each file
    4. Outputs results to specified directory

# Usage

This file will be used to batch run an entire database of files:

python batch_producer.py \
  --edf-directory data/edf_files \
  --output-directory output/processed \
  --producer-script producer/producer.py \
  --bootstrap-servers localhost:9092 \
  --window-size 4.0 \
  --step-size 2.0 \
  --batch-size 256

  Add --no-bandpower to disable power calculations.

# Output Files

[filename]_annotation_info.txt

    - Source file name, run, movements

    - Number and timing of annotations

JSON Output Example:

{
  "file_info": {
    "filename": "S001R03.edf",
    "sampling_frequency": 160,
    "duration_seconds": 64.0
  },
  "annotations_summary": {
    "total_found_in_edf": 3,
    "types_present": ["T0", "T1", "T2"]
  },
  "event_band_statistics_uv_sq": { ... },
  "producer_overall_avg_band_psd_uv_sq_hz": {
    "Alpha": 0.00042,
    "Beta": 0.00031
  }
}