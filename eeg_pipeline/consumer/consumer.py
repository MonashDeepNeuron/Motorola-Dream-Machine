import argparse
import json
from kafka import KafkaConsumer
from pydantic import ValidationError
from eeg_pipeline.schemas.eeg_schemas import EEGBatch


def consume_raw_eeg(bootstrap_servers: str, topic: str, group_id: str):
    """
    Consume EEGBatch messages from Kafka, validate with Pydantic, and print basic info.
    """
    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode('utf-8'))
    )

    print(f"Subscribed to topic '{topic}', waiting for messages...")
    for msg in consumer:
        data = msg.value
        if isinstance(data, str): 
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                print("Received non-JSON string, skipping...")
                continue
        try:
            batch = EEGBatch.model_validate(data)
            print(f"Received batch {batch.seq_number} from device {batch.device_id}, samples: {len(batch.data)}")
        except ValidationError as e:
            print(f"Validation error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kafka consumer for raw EEG batches")
    parser.add_argument('--bootstrap-servers', default='localhost:9092',
                        help='Kafka bootstrap servers')
    parser.add_argument('--topic', default='raw-eeg', help='Kafka topic to subscribe')
    parser.add_argument('--group-id', default='eeg-consumer-group', help='Consumer group ID')
    args = parser.parse_args()

    consume_raw_eeg(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id
    )