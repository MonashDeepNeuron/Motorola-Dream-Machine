import argparse
import json
import signal
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from kafka import KafkaConsumer, ConsumerRebalanceListener
from kafka.errors import KafkaError 
from pydantic import ValidationError

from eeg_pipeline.analysis.plotting import (
    plot_avg_band_power,
    plot_band_power_over_time,
)
from eeg_pipeline.schemas.eeg_schemas import EEGBatch, WindowBandPower

# --- Graceful shutdown state ---
stop_now = False
consumer_instance = None

def _sigint_handler(signum, frame):
    global stop_now, consumer_instance
    print("\nCtrl-C received. Initiating shutdown...")
    stop_now = True
    if consumer_instance:
        consumer_instance.wakeup()

signal.signal(signal.SIGINT, _sigint_handler)


class SimpleRebalanceListener(ConsumerRebalanceListener):
    def on_partitions_revoked(self, revoked):
        print(f"Partitions revoked: {revoked}")

    def on_partitions_assigned(self, assigned):
        print(f"Partitions assigned: {assigned}")


def main(
    bootstrap_servers: str,
    topic: str,
    group_id: str,
    write_json: bool,
    write_png: bool,
    window_size_s: float,
    step_size_s: float,
):
    global consumer_instance
    consumer_config = {
        'bootstrap_servers': bootstrap_servers,
        'group_id': group_id,
        'auto_offset_reset': 'earliest',
        'enable_auto_commit': True,
        'value_deserializer': lambda v: v.decode('utf-8', errors='ignore'),
        'consumer_timeout_ms': 5000,  # Increased timeout to 5 seconds
        'heartbeat_interval_ms': 1000,  # More frequent heartbeats
        'session_timeout_ms': 10000,    # Default is 10s, ensure it's > heartbeat
    }

    consumer = None
    window_band_channel_powers: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_band_values: Dict[str, List[float]] = defaultdict(list)
    processed_message_count = 0
    raw_batch_count = 0
    band_power_msg_count = 0
    loop_iterations = 0

    try:
        print(f"Attempting to connect to Kafka: {bootstrap_servers} with group ID: {group_id} for topic: {topic}")
        consumer = KafkaConsumer(topic, **consumer_config)
        consumer_instance = consumer
        print(f"Successfully subscribed to topic '{topic}'. Waiting for messages... (Ctrl-C to stop)")
    except KafkaError as e:
        print(f"CRITICAL: Failed to connect to Kafka or subscribe to topic: {e}")
        return

    try:
        print("Entering consumer message loop...")
        for msg in consumer:
            loop_iterations += 1
            if stop_now:
                print(f"DEBUG: stop_now flag is True at iteration {loop_iterations}. Breaking loop.")
                break
            
            processed_message_count += 1
            payload = msg.value

            try:
                batch = EEGBatch.model_validate_json(payload)
                raw_batch_count +=1
                print(
                    f"[RawEEG] Offset={msg.offset}, Seq={batch.seq_number}, "
                    f"Samples={len(batch.data)}, Chans={len(batch.channels)}, "
                    f"Labels={bool(batch.classification_labels)}"
                )
                continue
            except ValidationError:
                pass
            except json.JSONDecodeError:
                print(f"Error: Non-JSON message at offset {msg.offset} in topic '{topic}'. Skipping.")
                continue

            try:
                wb = WindowBandPower.model_validate_json(payload)
                band_power_msg_count +=1
                if wb.power and wb.channel_labels:
                    window_band_channel_powers[wb.window_index][wb.band].append(wb.power[0])
                    per_band_values[wb.band].append(wb.power[0])
                
                if band_power_msg_count % 100 == 0:
                    print(f"[BandPower] Aggregated {band_power_msg_count} messages. "
                          f"Current windows: {len(window_band_channel_powers)}")
                continue
            except ValidationError as e_val:
                is_json_error = any('JSONDecodeError' in err.get('type', '') for err in e_val.errors())
                if is_json_error:
                    print(f"Error: Malformed JSON for WindowBandPower at offset {msg.offset}. Skipping.")
                else:
                    print(f"Unrecognised/Invalid Pydantic schema for message at offset {msg.offset} in topic '{topic}'. Skipping.")
            except json.JSONDecodeError:
                print(f"Error: Malformed JSON for WindowBandPower at offset {msg.offset}. Skipping.")
                continue
            except Exception as e:
                print(f"Unexpected error processing message at offset {msg.offset}: {e}")

            if payload:
                print(f"Warning: Message at offset {msg.offset} in topic '{topic}' did not match known schemas and was not empty JSON.")

        if not stop_now:
            print(f"DEBUG: Consumer loop finished after {loop_iterations} iterations. stop_now is False.")

    except KeyboardInterrupt:
        print("\nDEBUG: KeyboardInterrupt caught in main loop. Shutting down...")
    except KafkaError as e:
        print(f"CRITICAL: KafkaError during consumption: {e}")
    except Exception as e:
        print(f"CRITICAL: Unexpected error in consumer loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if consumer:
            print("Closing Kafka consumer...")
            consumer.close()
        print(f"Total loop iterations: {loop_iterations}")
        print(f"Total messages processed from topic '{topic}': {processed_message_count}")
        print(f"Raw EEGBatch messages: {raw_batch_count}, WindowBandPower messages: {band_power_msg_count}")

    if band_power_msg_count > 0 and (write_json or write_png):
        output_dir = Path(".").resolve()
        print(f"\nWriting consumed band power artefacts to {output_dir}")

        final_window_band_avg: Dict[int, Dict[str, float]] = defaultdict(dict)
        for w_idx, bands_data in window_band_channel_powers.items():
            for band_name, S_powers in bands_data.items():
                final_window_band_avg[w_idx][band_name] = float(np.mean(S_powers)) if S_powers else 0.0
        
        overall_band_averages = {
            b: float(np.mean(vals)) if vals else 0.0 for b, vals in per_band_values.items()
        }

        if write_json:
            if final_window_band_avg:
                json_win_path = output_dir / "consumed_window_band_power_averages.json"
                windows_output_list = []
                for w_idx in sorted(final_window_band_avg.keys()):
                    win_data = {
                        "window_index": w_idx,
                        "start_time_sec": round(w_idx * step_size_s, 3),
                        "end_time_sec": round(w_idx * step_size_s + window_size_s, 3),
                        "band_power_avg_across_channels": {
                            band: round(power, 5) for band, power in final_window_band_avg[w_idx].items()
                        }
                    }
                    windows_output_list.append(win_data)
                
                with open(json_win_path, "w") as jf:
                    json.dump(
                        {
                            "metadata": {
                                "source_topic": topic,
                                "consumer_group_id": group_id,
                                "window_size_sec_config": window_size_s,
                                "step_size_sec_config": step_size_s,
                            },
                            "total_windows_processed": len(final_window_band_avg),
                            "window_data": windows_output_list,
                        },
                        jf,
                        indent=4,
                    )
                print(f"Saved consumed window band power averages to {json_win_path}")

            if overall_band_averages:
                json_overall_path = output_dir / "consumed_overall_band_averages.json"
                with open(json_overall_path, "w") as jf:
                    json.dump({"overall_average_band_power_consumed": overall_band_averages}, jf, indent=4)
                print(f"Saved consumed overall band averages to {json_overall_path}")
        
        if write_png:
            if overall_band_averages:
                plot_avg_band_power(
                    overall_band_averages,
                    output_dir / "consumed_average_band_power.png",
                    title="Average Consumed Band Power (Across Channels & Windows)"
                )
            if final_window_band_avg:
                plot_band_power_over_time(
                    final_window_band_avg,
                    step_size_s,
                    window_size_s,
                    output_dir / "consumed_window_band_power_over_time.png",
                    title="Consumed Band Power Over Time (Averaged Across Channels)"
                )
    elif (write_json or write_png):
        print("\nNo band power messages were consumed or aggregated, skipping artefact generation.")

    print("Consumer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka consumer for EEG pipeline data")
    parser.add_argument('--bootstrap-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', required=True, help='Kafka topic to subscribe to (e.g., raw-eeg or eeg-bandpower)')
    parser.add_argument('--group-id', default='eeg-consumer-group', help='Kafka consumer group ID')
    parser.add_argument('--write-json', action='store_true', help='Write aggregated band power data to JSON files on exit (if consuming eeg-bandpower)')
    parser.add_argument('--write-png', action='store_true', help='Generate and save band power plots on exit (if consuming eeg-bandpower)')
    parser.add_argument('--window-size', dest='window_size_s', type=float, default=4.0, help='Window size in seconds (for metadata in output files if --write-json)')
    parser.add_argument('--step-size', dest='step_size_s', type=float, default=2.0, help='Step size in seconds (for metadata in output files if --write-json)')
    
    args = parser.parse_args()

    main(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        group_id=args.group_id,
        write_json=args.write_json,
        write_png=args.write_png,
        window_size_s=args.window_size_s,
        step_size_s=args.step_size_s,
    )