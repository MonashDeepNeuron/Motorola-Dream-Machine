import argparse
import json
from uuid import uuid4
from datetime import datetime, timedelta

import mne
from kafka import KafkaProducer
from eeg_pipeline.schemas.eeg_schemas import EEGBatch


def read_edf(edf_path: str):
    """
    Load EDF file and return raw data and channel names.
    Returns:
        sample_rate (float), channel_names (List[str]), data (2D List[float]), timestamps (List[datetime])
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data_np = raw.get_data()  # shape (n_channels, n_samples)

    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    n_samples = data_np.shape[1]

    start_time = raw.info.get('meas_date') or datetime.utcnow()
    timestamps = [start_time + timedelta(seconds=i / sfreq) for i in range(n_samples)]

    # [batch_size x n_channels]
    data_list = data_np.T.tolist()
    return sfreq, ch_names, data_list, timestamps


def produce_batches(edf_path: str, bootstrap_servers: str, topic: str, batch_size: int):
    # Read EDF
    sfreq, channels, data_list, timestamps = read_edf(edf_path)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    session_id = uuid4()
    total_samples = len(data_list)
    seq_num = 0

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        batch_data = data_list[start_idx:end_idx]
        batch_t0 = timestamps[start_idx]

        batch = EEGBatch(
            device_id="producer",  
            session_id=session_id,
            timestamp=batch_t0,
            seq_number=seq_num,
            sample_rate=sfreq,
            channels=channels,
            data=batch_data
        )

        # Dump to dict for serialization
        payload_dict = batch.model_dump_json()
        producer.send(topic, payload_dict)
        print(f"Sent batch {seq_num} ({start_idx}:{end_idx}) to topic '{topic}'")

        seq_num += 1

    producer.flush()
    print("All batches sent.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EDF to Kafka EEG Batch Producer")
    parser.add_argument('--edf-file', required=True, help='Path to EDF file')
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='raw-eeg', help='Kafka topic to publish')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Number of samples per batch')
    args = parser.parse_args()

    produce_batches(
        edf_path=args.edf_file,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        batch_size=args.batch_size
    )
