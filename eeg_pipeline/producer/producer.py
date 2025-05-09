import argparse
import json
from typing import List
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import mne
from kafka import KafkaProducer
from eeg_pipeline.schemas.eeg_schemas import EEGBatch

def load_and_prepare_labels(raw: mne.io.Raw,
                            total_samples: int) -> List[str]:
    """
    Extracts annotations from the `.edf.event` raw object and creates per-sample labels.
    Returns:
        List[str]: A list of labels, one for each sample. Defaults to'background'.
    """
    sample_labels = ['background'] * total_samples

    if not raw.annotations or len(raw.annotations) == 0:
        print("No MNE annotations found in the Raw object. All samples will be labeled 'background'.")
        return sample_labels

    sfreq = raw.info['sfreq']
    print(f"Found {len(raw.annotations)} MNE annotations. Processing...")
    for annot in raw.annotations:
        onset_sec = annot['onset']
        duration_sec = annot['duration']
        description = annot['description']

        start_sample = int(onset_sec * sfreq)
        num_samples_in_event = int(duration_sec * sfreq)
        if num_samples_in_event == 0 and duration_sec > 0: # if duration is very small but non-zero
             num_samples_in_event = 1
        elif duration_sec == 0: 
            num_samples_in_event = 1

        end_sample = start_sample + num_samples_in_event

        for s_idx in range(start_sample, min(end_sample, total_samples)):
            if 0 <= s_idx < total_samples:
                sample_labels[s_idx] = description

    unique_labels_found = set(sample_labels)
    print(f"Finished processing MNE annotations. Unique labels generated: {unique_labels_found}")
    return sample_labels

def read_edf(edf_path: str):
    """
    Load EDF file and return raw data and channel names.
    Returns:
        raw (mne.io.Raw), sample_rate (float), channel_names (List[str]), data (2D List[float]), total_samples (int)
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True)
    data_np = raw.get_data()  # shape (n_channels, n_samples)

    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    total_samples = data_np.shape[1]

    start_time = raw.info.get('meas_date')
    if start_time is None:
        start_time = datetime.now(timezone.utc)
    elif start_time.tzinfo is None:  
        start_time = start_time.replace(tzinfo=timezone.utc)
    
    timestamps = [start_time + timedelta(seconds=i / sfreq) for i in range(total_samples)]

    # [batch_size x n_channels]
    data_list = data_np.T.tolist()
    return raw, sfreq, ch_names, data_list, timestamps, total_samples


def produce_batches(edf_path: str, bootstrap_servers: str, topic: str, batch_size: int):
    # Read EDF
    raw, sfreq, channels, data_list, timestamps, total_samples = read_edf(edf_path)
    all_classification_labels = load_and_prepare_labels(raw, total_samples)
    
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v_str: v_str.encode('utf-8')
    )

    session_id = uuid4()
    seq_num = 0
    processed_samples_count = 0

    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        if start_idx == end_idx:
            continue
            
        batch_data = data_list[start_idx:end_idx]
        batch_t0 = timestamps[start_idx]
        batch_labels = all_classification_labels[start_idx:end_idx]

        batch = EEGBatch(
            device_id="producer",  
            session_id=session_id,
            timestamp=batch_t0,
            seq_number=seq_num,
            sample_rate=sfreq,
            channels=channels,
            data=batch_data,
            classification_labels=batch_labels
        )

        # Dump to dict for serialization
        payload_str = batch.model_dump_json()
        producer.send(topic, payload_str)
        print(f"Sent batch {seq_num} ({start_idx}:{end_idx}) to topic '{topic}'")

        seq_num += 1
        processed_samples_count += len(batch_data)

    producer.flush()
    print(f"All batches sent. Total batches: {seq_num}, Total samples: {processed_samples_count}/{total_samples}")


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
