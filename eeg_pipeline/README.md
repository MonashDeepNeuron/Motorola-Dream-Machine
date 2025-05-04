A lightweight end‑to‑end Kafka‑based pipeline for ingesting, validating, and relaying EEG data batches from EDF files to downstream consumers (inference device)

- Reads EDF files via MNE, batches samples into EEGBatch messages, validates with Pydantic, and publishes to Kafka
- Subscribes to raw‑EEG topic, validates incoming JSON against the same schema, and prints or forwards to inference services

```bash
pip install kafka-python mne pydantic
```

```bash
cd config
docker-compose up -d
```

- Zookeeper on port 2181
- Kafka on port 9092 (advertised as localhost:9092)
- Schema Registry on port 8081

-> Create kafka topics

```bash
docker exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic raw-eeg --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 \
  --create --topic eeg-bandpower --partitions 3 --replication-factor 1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```

So, just run the consumer

```bash
python -m consumer.consumer \
  --bootstrap-servers localhost:9092 \
  --topic raw-eeg \
  --group-id my-listener
```

And then for producer with relative path:

```bash
python -m producer.producer \
  --edf-file {path_to_eds_file}.edf \
  --bootstrap-servers localhost:9092 \
  --topic raw-eeg \
  --batch-size 256 # number of samples per message (e.g. 256 samples at 256Hz = 1s of data) \
```

---

# Input and Output Formats

## Producer

- CLI arguements

  - `--edf-file` (required): Path to the EDF file
  - `bootstrap-servers` (default: localhost://9092): Kafka brokers
  - `--topic` (default == `raw-eeg`): Kafka topic to publish
  - `--batch-size` (default = 256) : Number of samples per batch

- Function Signature

```py
produce_batches(
  edf_path: str,
  bootstrap_servers: str,
  topic: str,
  batch_size: int
)
```

- Kafka message (JSON) Output format

```py
{
  "device_id": "producer",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",  // UUID string
  "timestamp": "2025-05-05T04:00:00Z",
  "seq_number": 0,
  "sample_rate": 256.0,
  "channels": ["Fz","Cz","Pz",…],
  "data": [
    [0.12, -0.03, 0.08, …],
    [0.11, -0.02, 0.09, …],
    …
  ]
}
```

- Producer Console Output

```py
Sent batch 0 (0:256) to topic 'raw-eeg'
Sent batch 1 (256:512) to topic 'raw-eeg'
...
All batches sent.
```

---

## Consumer

- CLI arguements

  - `bootstrap-servers` (default: localhost://9092): Kafka brokers
  - `--topic` (default == `raw-eeg`): Kafka topic to subscribe to
  - `--group-id` (default = `eeg-consumer-group`) : Consumer group ID

- Function Signature

```py
consume_raw_eeg(
  bootstrap_servers: str,
  topic: str,
  group_id: str
)
```

- Kafka message (JSON) Input format

```py
{
  "device_id": "producer",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",  // UUID string
  "timestamp": "2025-05-05T04:00:00Z",
  "seq_number": 0,
  "sample_rate": 256.0,
  "channels": ["Fz","Cz","Pz",…],
  "data": [
    [0.12, -0.03, 0.08, …],
    [0.11, -0.02, 0.09, …],
    …
  ]
}
```

- Consumer Console Output

```py
Subscribed to topic 'raw-eeg', waiting for messages... # On startup
Received batch 3 from device producer, samples: 256 # On valid batc
Received non-JSON string, skipping... # On non-JSON payload

Validation error: 1 validation error for EEGBatch # On Schema validation
data -> 5 -> <root>: Each row in data must have len equal to num of channels
...
All batches sent.
```
