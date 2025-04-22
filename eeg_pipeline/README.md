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
