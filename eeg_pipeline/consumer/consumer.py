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
        value_deserializer=lambda v: v.decode('utf-8')
    )

    print(f"Subscribed to topic '{topic}', waiting for messages...")
    # msg_count = 0
    for msg in consumer:
        # msg_count += 1
        try:
            batch = EEGBatch.model_validate_json(msg.value)
            # if msg_count % 10 == 0 or msg_count == 1: 
            print(f"\nReceived batch {batch.seq_number} (Kafka offset: {msg.offset}) from device {batch.device_id}")
            print(f"  Timestamp: {batch.timestamp}, Samples: {len(batch.data)}, Channels: {len(batch.channels)}")
            
            if batch.classification_labels:
                unique_labels_in_batch = set(batch.classification_labels)
                print(f"  Classification labels present. Unique in batch: {unique_labels_in_batch}. First 5: {batch.classification_labels[:5]}")
            else:
                print("  No classification labels in this batch.")

            # -- Here we would forward to an inference device -- 

        except json.JSONDecodeError:
            print(f"Error: Received non-JSON string or malformed JSON (offset {msg.offset}), skipping: {msg.value[:200]}...")
            continue
        except ValidationError as e:
            print(f"Error: EEGBatch Schema validation failed (offset {msg.offset}):")
            for error in e.errors():
                print(f"  Field: {'.'.join(map(str,error['loc']))}, Message: {error['msg']}, Input: {error['input']}")
        except Exception as e:
            print(f"An unexpected error occurred while processing message (offset {msg.offset}): {e}")

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