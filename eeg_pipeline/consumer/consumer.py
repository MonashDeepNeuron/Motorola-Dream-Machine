import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import signal
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Any, Deque, Optional

import numpy as np
from kafka import KafkaConsumer, ConsumerRebalanceListener
from kafka.errors import KafkaError
from pydantic import ValidationError

from analysis.plotting import (
    plot_avg_band_power,
    plot_band_power_over_time,
)
from schemas.eeg_schemas import WindowBandPower, EEGSample
from analysis import calculate_psd_for_window 


WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0

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
        "bootstrap_servers": bootstrap_servers,
        "group_id": group_id,
        "auto_offset_reset": "earliest",
        "enable_auto_commit": True,
        "value_deserializer": lambda v: v.decode("utf-8", errors="ignore"),
        "heartbeat_interval_ms": 1000,  # More frequent heartbeats
        "session_timeout_ms": 10000,  # Default is 10s, ensure it's > heartbeat
    }

    consumer = None
    processed_message_count = 0
    band_power_msg_count = 0
    loop_iterations = 0

    # --- State for real-time analysis ---
    buffer: Deque[List[float]] = deque()
    sample_rate: Optional[float] = None
    window_size_samples: Optional[int] = None
    step_size_samples: Optional[int] = None
    window_index = 0

    # --- Storage for final report ---
    # This replaces the old way of collecting WindowBandPower messages
    final_window_band_avg: Dict[int, Dict[str, float]] = defaultdict(dict)
    overall_band_averages_collector: Dict[str, List[float]] = defaultdict(list)

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

            try:
                sample = EEGSample.model_validate_json(msg.value)
                processed_message_count += 1

                if sample_rate is None:
                    print(f"First message received. Setting up parameters with sample_rate={sample.sample_rate} Hz.")
                    sample_rate = sample.sample_rate
                    window_size_samples = int(window_size_s * sample_rate)
                    step_size_samples = int(step_size_s * sample_rate)
                    # Set maxlen to avoid unbounded memory usage
                    buffer = deque(maxlen=window_size_samples + step_size_samples)

                buffer.append(sample.sample_data)

                if len(buffer) >= window_size_samples:
                    window_np = np.asarray(list(buffer)[-window_size_samples:]) 

                    psd_results = calculate_psd_for_window(window_np, sample_rate)

                    final_window_band_avg[window_index] = psd_results
                    for band, power in psd_results.items():
                        overall_band_averages_collector[band].append(power)

                    print(f"[Analysis] Window {window_index}: Alpha Power = {psd_results.get('alpha', 0.0):.2f} μV²/Hz")
                    window_index += 1

                    for _ in range(step_size_samples):
                        if buffer:
                            buffer.popleft()

                continue  # Move to the next message

            except (ValidationError, json.JSONDecodeError):
                pass

            try:
                wb = WindowBandPower.model_validate_json(msg.value)
                band_power_msg_count += 1
                if wb.power and wb.channel_labels:
                    pass
                if band_power_msg_count % 100 == 0:
                     print(f"[BandPower Fallback] Received {band_power_msg_count} messages.")
                continue
            except (ValidationError, json.JSONDecodeError):
                pass

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
        print(f"Total EEGSample messages processed: {processed_message_count}")
        if band_power_msg_count > 0:
            print(f"Total WindowBandPower messages (fallback): {band_power_msg_count}")

    # *** IMPORTANT: UPDATE THE OUTPUT CALCULATION ***
    # This now calculates the overall averages from the data we collected
    overall_band_averages = {
        band: float(np.mean(powers)) if powers else 0.0
        for band, powers in overall_band_averages_collector.items()
    }

    if final_window_band_avg and (write_json or write_png):
        output_dir = Path(".").resolve()
        print(f"\nWriting consumed band power artefacts to {output_dir}")

        if write_json:
            json_win_path = output_dir / "consumed_window_band_power_averages.json"
            windows_output_list = []
            for w_idx in sorted(final_window_band_avg.keys()):
                win_data = {
                    "window_index": w_idx,
                    "start_time_sec": round(w_idx * step_size_s, 3),
                    "end_time_sec": round(w_idx * step_size_s + window_size_s, 3),
                    "band_power_avg_across_channels": {
                        band: round(power, 5) for band, power in final_window_band_avg[w_idx].items()
                    },
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
                    title="Average Consumed Band Power (Across Channels & Windows)",
                )
            if final_window_band_avg:
                plot_band_power_over_time(
                    final_window_band_avg,
                    step_size_s,
                    window_size_s,
                    output_dir / "consumed_window_band_power_over_time.png",
                    title="Consumed Band Power Over Time (Averaged Across Channels)",
                )
    elif write_json or write_png:
        print("\nNo analysis windows were processed, skipping artefact generation.")

    print("Consumer finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka consumer for EEG pipeline data")
    parser.add_argument("--bootstrap-servers", default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", required=True, help="Kafka topic to subscribe to (e.g., raw-eeg)")
    parser.add_argument("--group-id", default="eeg-consumer-group", help="Kafka consumer group ID")
    parser.add_argument("--write-json", action="store_true", help="Write aggregated band power data to JSON files on exit")
    parser.add_argument("--write-png", action="store_true", help="Generate and save band power plots on exit")
    parser.add_argument("--window-size", dest="window_size_s", type=float, default=4.0, help="Window size in seconds for analysis")
    parser.add_argument("--step-size", dest="step_size_s", type=float, default=2.0, help="Step size in seconds for analysis")
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
