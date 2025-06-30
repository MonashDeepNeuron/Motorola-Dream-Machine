from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import mne
import numpy as np
from kafka import KafkaProducer
from kafka.errors import KafkaError

import sys

# Ensure the repo root is on sys.path so eeg_pipeline can be imported
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
sys.path.insert(0, project_root)


from eeg_pipeline.analysis.bands import compute_window_band_power, DefaultBands, WindowBandResult
from eeg_pipeline.analysis.header import header_dump
from eeg_pipeline.analysis.events import event_band_statistics
from eeg_pipeline.schemas.eeg_schemas import EEGBatch, WindowBandPower
from collections import defaultdict

def get_run_specific_movements(edf_basename: str) -> Tuple[str, str, Optional[str]]:
    t1_movement = "Unknown"
    t2_movement = "Unknown"
    run_number_str = None
    try:
        parts = edf_basename.upper().split('R')
        if len(parts) > 1:
            run_part = parts[1].split('.')[0]
            if run_part.isdigit():
                run_number_str = run_part
                run_number = int(run_part)
                if run_number in [3, 4, 7, 8, 11, 12]:
                    t1_movement = "Left Fist"
                    t2_movement = "Right Fist"
                elif run_number in [5, 6, 9, 10, 13, 14]:
                    t1_movement = "Both Fists"
                    t2_movement = "Both Feet"
                elif run_number in [1, 2]:
                    t1_movement = "Rest (Baseline)"
                    t2_movement = "Rest (Baseline)"
    except Exception as e:
        print(f"Could not determine run number or movements from filename '{edf_basename}': {e}")
    return t1_movement, t2_movement, run_number_str

def load_and_prepare_labels(
    raw: mne.io.Raw,
    total_samples: int,
    t1_movement: str,
    t2_movement: str
) -> Tuple[List[str], List[Dict[str, Any]]]:
    sample_labels = ['background'] * total_samples
    annotation_details_for_report: List[Dict[str, Any]] = []
    event_map_for_report = {"T0": "Rest", "T1": t1_movement, "T2": t2_movement}

    if not raw.annotations or len(raw.annotations) == 0:
        print("No MNE annotations found in the Raw object. All samples will be labeled 'background'.")
        return sample_labels, annotation_details_for_report

    sfreq = raw.info['sfreq']
    print(f"Found {len(raw.annotations)} MNE annotations. Processing for per-sample labels...")
    for i, annot in enumerate(raw.annotations):
        onset_sec = annot['onset']
        duration_sec = annot['duration']
        description = annot['description']

        start_sample = int(onset_sec * sfreq)
        num_samples_in_event = max(1, int(duration_sec * sfreq)) if duration_sec > 0 else 1
        end_sample = start_sample + num_samples_in_event

        label_to_assign = description
        for s_idx in range(start_sample, min(end_sample, total_samples)):
            if 0 <= s_idx < total_samples:
                sample_labels[s_idx] = label_to_assign

        annotation_details_for_report.append({
            "index": i,
            "type": description,
            "movement": event_map_for_report.get(description, "Unknown Event Type"),
            "onset_sec": round(onset_sec, 3),
            "duration_sec": round(duration_sec, 3)
        })

    unique_labels_found = sorted(list(set(sample_labels)))
    print(f"Finished processing MNE annotations. Unique sample labels generated: {unique_labels_found}")
    return sample_labels, annotation_details_for_report

def read_edf_file(edf_path: str):
    print(f"Loading EDF file: {edf_path}...")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='WARNING')
    raw.rename_channels(lambda x: x.strip('. ').replace('..', '').upper())
    data_np_volts = raw.get_data(units='V')
    sfreq = raw.info['sfreq']
    ch_names = raw.ch_names
    total_samples = data_np_volts.shape[1]

    start_datetime_obj = raw.info.get('meas_date')
    if start_datetime_obj is None:
        start_datetime_obj = datetime.now(timezone.utc)
    elif start_datetime_obj.tzinfo is None:
        start_datetime_obj = start_datetime_obj.replace(tzinfo=timezone.utc)
    sample_timestamps = [start_datetime_obj + timedelta(seconds=i / sfreq) for i in range(total_samples)]
    data_list_volts = data_np_volts.T.tolist()
    print(f"EDF loaded: {total_samples} samples, {len(ch_names)} channels, {sfreq} Hz.")
    return raw, sfreq, ch_names, data_list_volts, sample_timestamps, total_samples

def produce_messages(
    edf_path_str: str,
    bootstrap_servers: str,
    raw_eeg_topic: str,
    batch_size: int,
    emit_bandpower: bool,
    bandpower_topic: str,
    window_size_s: float,
    step_size_s: float,
):
    edf_path = Path(edf_path_str)
    edf_basename = edf_path.name

    try:
        raw, sfreq, channels, data_list_volts, sample_timestamps, total_samples = read_edf_file(edf_path_str)
    except Exception as e:
        print(f"Error reading EDF file {edf_path_str}: {e}")
        return

    print("\n--- EDF Header (from MNE) ---")
    for line in header_dump(raw):
        print(line)
    print("--- End of EDF Header ---\n")

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v_json: json.dumps(v_json).encode('utf-8'),
            acks='all',
            retries=3,
            retry_backoff_ms=1000,
        )
    except KafkaError as e:
        print(f"Error creating Kafka producer: {e}")
        return

    session_id = uuid4()
    t1_movement, t2_movement, run_number_str = get_run_specific_movements(edf_basename)
    print(f"Determined movements for run '{run_number_str}': T1='{t1_movement}', T2='{t2_movement}'")

    all_classification_labels, annotation_details_for_report = load_and_prepare_labels(
        raw, total_samples, t1_movement, t2_movement
    )

    per_window_channel_band_results: List[WindowBandResult] = []
    producer_window_band_avg_across_channels: Dict[int, Dict[str, float]] = {}

    if emit_bandpower:
        print(f"\nComputing windowed band power (window: {window_size_s}s, step: {step_size_s}s)...")
        per_window_channel_band_results, producer_window_band_avg_across_channels = compute_window_band_power(
            raw,
            window_size_s=window_size_s,
            step_size_s=step_size_s,
            bands=DefaultBands,
        )
        band_power_messages_sent = 0
        for res in per_window_channel_band_results:
            wb_message = WindowBandPower(
                device_id="producer",
                session_id=session_id,
                window_index=res.window_index,
                start_time_sec=res.start_sec,
                end_time_sec=res.end_sec,
                band=res.band_name,
                channel_labels=[res.channel_label],
                power=[res.power],
            )
            try:
                producer.send(bandpower_topic, wb_message.model_dump(mode='json'))
                band_power_messages_sent += 1
            except Exception as e:
                print(f"Error sending WindowBandPower message: {e}")

        producer.flush()
        print(f"Emitted {band_power_messages_sent} WindowBandPower messages to topic '{bandpower_topic}'.")

    print(f"\nProducing EEGBatch messages to topic '{raw_eeg_topic}'...")
    seq_num = 0
    raw_batches_sent = 0
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        if start_idx == end_idx:
            continue

        batch_data_volts = [data_list_volts[i] for i in range(start_idx, end_idx)]
        batch_timestamp = sample_timestamps[start_idx]
        batch_labels = all_classification_labels[start_idx:end_idx] if all_classification_labels else None

        eeg_batch_msg = EEGBatch(
            device_id="producer",
            session_id=session_id,
            timestamp=batch_timestamp,
            seq_number=seq_num,
            sample_rate=sfreq,
            channels=channels,
            data=batch_data_volts,
            classification_labels=batch_labels,
        )
        try:
            producer.send(raw_eeg_topic, eeg_batch_msg.model_dump(mode='json'))
            raw_batches_sent += 1
            if seq_num % 50 == 0 or seq_num < 5:
                print(f"Sent EEGBatch {seq_num} ({start_idx}:{end_idx}, {len(batch_data_volts)} samples) to topic '{raw_eeg_topic}'")
        except Exception as e:
            print(f"Error sending EEGBatch message for seq {seq_num}: {e}")

        seq_num += 1

    producer.flush()
    print(f"All {raw_batches_sent} EEGBatch messages sent for this EDF file.")

    print("\nPerforming final analysis for local reports...")
    event_stats = event_band_statistics(raw, DefaultBands)
    print("Event-band statistics computed.")

    overall_avg_psd_producer: Dict[str, float] = {}
    if producer_window_band_avg_across_channels:
        temp_band_powers: Dict[str, List[float]] = defaultdict(list)
        for _win_idx, band_data in producer_window_band_avg_across_channels.items():
            for band_name, power_val in band_data.items():
                temp_band_powers[band_name].append(power_val)
        for band_name, powers in temp_band_powers.items():
            overall_avg_psd_producer[band_name] = float(np.mean(powers)) if powers else 0.0

    output_dir = Path(".").resolve()
    annotation_txt_path = output_dir / f"{edf_basename}_annotation_info.txt"
    with open(annotation_txt_path, 'w') as f:
        f.write("ANNOTATION INFORMATION\n")
        f.write("=====================\n\n")
        f.write(f"Source EDF: {edf_basename}\n")
        f.write(f"Run Number: {run_number_str if run_number_str else 'Unknown'}\n")
        f.write(f"T0 Interpretation: Rest\n")
        f.write(f"T1 Interpretation: {t1_movement}\n")
        f.write(f"T2 Interpretation: {t2_movement}\n\n")
        f.write(f"Total MNE Annotations Found: {len(raw.annotations)}\n")
        unique_event_types = sorted(list(set(annot['type'] for annot in annotation_details_for_report)))
        f.write(f"Annotation Types Present: {', '.join(unique_event_types)}\n\n")
        f.write("Detailed Annotation Timing:\n")
        f.write("-------------------------\n")
        for annot_item in annotation_details_for_report:
            f.write(
                f"{annot_item['index'] + 1}. Type='{annot_item['type']}' ({annot_item['movement']}): "
                f"Onset at {annot_item['onset_sec']:.2f}s, Duration: {annot_item['duration_sec']:.2f}s\n"
            )
    print(f"Saved annotation information to: {annotation_txt_path}")

    results_json_path = output_dir / f"{edf_basename}_eeg_analysis_results.json"
    final_results_data = {
        "file_info": {
            "filename": edf_basename,
            "run_number": run_number_str if run_number_str else "unknown",
            "sampling_frequency": sfreq,
            "t1_movement_interpretation": t1_movement,
            "t2_movement_interpretation": t2_movement,
            "total_samples": total_samples,
            "duration_seconds": round(total_samples / sfreq, 2)
        },
        "annotations_summary": {
            "total_found_in_edf": len(raw.annotations),
            "types_present": unique_event_types,
            "details": annotation_details_for_report
        },
        "event_band_statistics_uv_sq": event_stats,
        "producer_overall_avg_band_psd_uv_sq_hz": overall_avg_psd_producer if overall_avg_psd_producer else "Not computed (emit_bandpower was false)"
    }
    with open(results_json_path, 'w') as f:
        json.dump(final_results_data, f, indent=4)
    print(f"Saved comprehensive EEG analysis results to: {results_json_path}")

    print(f"\nProducer finished processing {edf_basename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDF to Kafka Producer for EEG data and band power")
    parser.add_argument('--edf-file', required=True, help='Path to the EDF file')
    parser.add_argument('--bootstrap-servers', default='localhost:9092', help='Kafka bootstrap servers')
    parser.add_argument('--topic', dest='raw_eeg_topic', default='raw-eeg', help='Kafka topic for raw EEG batches')
    parser.add_argument('--batch-size', type=int, default=256, help='Number of samples per EEGBatch message')
    parser.add_argument('--emit-bandpower', action='store_true', help='Compute and publish per-window band power')
    parser.add_argument('--bandpower-topic', default='eeg-bandpower', help='Kafka topic for WindowBandPower messages')
    parser.add_argument('--window-size', dest='window_size_s', type=float, default=4.0, help='Window size in seconds for band power calculation')
    parser.add_argument('--step-size', dest='step_size_s', type=float, default=2.0, help='Step size in seconds for band power calculation')
    args = parser.parse_args()

    produce_messages(
        edf_path_str=args.edf_file,
        bootstrap_servers=args.bootstrap_servers,
        raw_eeg_topic=args.raw_eeg_topic,
        batch_size=args.batch_size,
        emit_bandpower=args.emit_bandpower,
        bandpower_topic=args.bandpower_topic,
        window_size_s=args.window_size_s,
        step_size_s=args.step_size_s,
    )
