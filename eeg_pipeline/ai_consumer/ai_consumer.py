#!/usr/bin/env python3
"""
AI-Powered EEG Consumer
Real-time inference from EEG signals to robot commands using the EEG2Arm model.
"""

import sys
import os
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import signal
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Deque

import numpy as np
import torch
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError
from pydantic import ValidationError

from schemas.eeg_schemas import EEGSample
from model.eeg_model import EEG2Arm
from analysis.bands import calculate_psd_for_window, DefaultBands

# Command mapping
COMMANDS = {
    0: "REST",
    1: "LEFT",
    2: "RIGHT",
    3: "FORWARD",
    4: "BACKWARD"
}

stop_now = False
consumer_instance = None


def _sigint_handler(signum, frame):
    global stop_now, consumer_instance
    print("\nShutdown initiated...")
    stop_now = True
    if consumer_instance:
        consumer_instance.wakeup()


signal.signal(signal.SIGINT, _sigint_handler)


class EEGToRobotInference:
    """
    Real-time EEG to robot command inference system.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_channels: int = 32,
        n_bands: int = 5,
        window_size_s: float = 4.0,
        step_size_s: float = 2.0,
        device: str = "cpu"
    ):
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.window_size_s = window_size_s
        self.step_size_s = step_size_s
        self.device = torch.device(device)
        
        # Initialize model
        print(f"Initializing EEG2Arm model ({n_channels} channels, {n_bands} bands)...")
        self.model = EEG2Arm(
            n_elec=n_channels,
            n_bands=n_bands,
            clip_length=None,  # Variable length
            n_classes=5,
            pointwise_groups=1,
            edges=self._create_electrode_edges()
        ).to(self.device)
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            print(f"Loading model weights from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            print("⚠️  WARNING: No trained model found. Using random weights.")
            print("   Model will make random predictions until trained.")
        
        self.model.eval()
        
        # Buffers
        self.sample_buffer: Deque[EEGSample] = deque()
        self.sample_rate: Optional[float] = None
        self.window_size_samples: Optional[int] = None
        self.step_size_samples: Optional[int] = None
        
        # Statistics
        self.predictions_made = 0
        self.last_prediction_time = time.time()
        
    def _create_electrode_edges(self) -> List[tuple]:
        """
        Create electrode adjacency graph based on 10-20 system.
        For 32 channels, creates a reasonable spatial connectivity.
        """
        # Simple ring topology as baseline
        edges = [(i, (i + 1) % self.n_channels) for i in range(self.n_channels)]
        
        # Add some cross connections for better spatial modeling
        for i in range(self.n_channels // 2):
            edges.append((i, i + self.n_channels // 2))
        
        return edges
    
    def preprocess_window(self, samples: List[EEGSample]) -> torch.Tensor:
        """
        Convert list of EEGSamples to model input format.
        
        Input: List of EEGSample objects
        Output: Tensor of shape (1, n_channels, n_bands, n_frames)
        """
        # Extract raw data
        raw_data = np.array([s.sample_data for s in samples])  # (n_samples, n_channels)
        raw_data = raw_data.T  # (n_channels, n_samples)
        
        # Compute band power for each time frame using sliding window
        # For simplicity, we'll compute one band power per sample
        # In practice, you might want overlapping windows
        
        n_frames = len(samples)
        band_powers = np.zeros((self.n_channels, self.n_bands, n_frames))
        
        # Use a small window for each frame (e.g., 0.5 seconds)
        frame_window_samples = min(int(0.5 * self.sample_rate), len(samples))
        
        for frame_idx in range(n_frames):
            start_idx = max(0, frame_idx - frame_window_samples // 2)
            end_idx = min(len(samples), frame_idx + frame_window_samples // 2)
            
            frame_data = raw_data[:, start_idx:end_idx]  # (n_channels, window_samples)
            
            # Compute PSD for this frame
            for ch_idx in range(self.n_channels):
                if frame_data.shape[1] < 10:  # Need minimum samples
                    continue
                
                # Simple band power calculation per channel
                # In production, use more sophisticated methods
                from scipy import signal
                freqs, psd = signal.welch(
                    frame_data[ch_idx],
                    fs=self.sample_rate,
                    nperseg=min(256, frame_data.shape[1]),
                    scaling='density'
                )
                
                # Extract band powers
                band_idx = 0
                for band_name, (fmin, fmax) in DefaultBands.items():
                    idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                    if np.any(idx_band):
                        power = np.mean(psd[idx_band]) * 1e12  # Convert to μV²/Hz
                        band_powers[ch_idx, band_idx, frame_idx] = power
                    band_idx += 1
        
        # Convert to tensor: (1, n_channels, n_bands, n_frames)
        input_tensor = torch.from_numpy(band_powers).float().unsqueeze(0)
        
        return input_tensor.to(self.device)
    
    def predict(self, samples: List[EEGSample]) -> Dict:
        """
        Make a prediction from a window of EEG samples.
        
        Returns:
            dict with 'command', 'confidence', 'probabilities'
        """
        # Preprocess
        model_input = self.preprocess_window(samples)
        
        # Inference
        with torch.no_grad():
            logits = self.model(model_input)  # (1, n_classes)
            probabilities = torch.softmax(logits, dim=1)[0]  # (n_classes,)
            
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        # Map to command
        command = COMMANDS.get(predicted_class, "UNKNOWN")
        
        self.predictions_made += 1
        current_time = time.time()
        inference_rate = 1.0 / (current_time - self.last_prediction_time + 1e-6)
        self.last_prediction_time = current_time
        
        return {
            "command": command,
            "command_id": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities.cpu().numpy().tolist(),
            "inference_rate_hz": inference_rate,
            "timestamp": current_time
        }
    
    def process_sample(self, sample: EEGSample) -> Optional[Dict]:
        """
        Add sample to buffer and make prediction if window is ready.
        """
        # Initialize parameters from first sample
        if self.sample_rate is None:
            self.sample_rate = sample.sample_rate
            self.window_size_samples = int(self.window_size_s * self.sample_rate)
            self.step_size_samples = int(self.step_size_s * self.sample_rate)
            print(f"Configured: {self.sample_rate} Hz, "
                  f"{self.window_size_samples} sample window, "
                  f"{self.step_size_samples} sample step")
        
        # Add to buffer
        self.sample_buffer.append(sample)
        
        # Check if we have enough samples for a window
        if len(self.sample_buffer) >= self.window_size_samples:
            # Get the last window_size_samples
            window_samples = list(self.sample_buffer)[-self.window_size_samples:]
            
            # Make prediction
            prediction = self.predict(window_samples)
            
            # Step forward by step_size_samples
            for _ in range(min(self.step_size_samples, len(self.sample_buffer))):
                if len(self.sample_buffer) > self.window_size_samples:
                    self.sample_buffer.popleft()
            
            return prediction
        
        return None


def main(
    kafka_servers: str,
    input_topic: str,
    output_topic: str,
    group_id: str,
    model_path: Optional[str],
    n_channels: int,
    device: str,
    log_file: Optional[str]
):
    global consumer_instance
    
    # Initialize AI inference engine
    print("=" * 60)
    print("AI-Powered EEG Consumer")
    print("=" * 60)
    
    inference_engine = EEGToRobotInference(
        model_path=model_path,
        n_channels=n_channels,
        device=device
    )
    
    # Setup Kafka consumer
    print(f"\nConnecting to Kafka: {kafka_servers}")
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=kafka_servers,
        group_id=group_id,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: v.decode("utf-8", errors="ignore")
    )
    consumer_instance = consumer
    
    # Setup Kafka producer for robot commands
    producer = KafkaProducer(
        bootstrap_servers=kafka_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        compression_type='lz4'
    )
    
    # Setup logging
    log_f = None
    if log_file:
        log_f = open(log_file, 'a')
        print(f"Logging predictions to {log_file}")
    
    print(f"Consuming from topic: {input_topic}")
    print(f"Producing to topic: {output_topic}")
    print("Waiting for EEG data...\n")
    
    try:
        for msg in consumer:
            if stop_now:
                break
            
            # Parse EEG sample
            try:
                sample = EEGSample.model_validate_json(msg.value)
            except (ValidationError, json.JSONDecodeError):
                continue
            
            # Process sample and get prediction
            prediction = inference_engine.process_sample(sample)
            
            if prediction:
                # Log prediction
                print(f"[Prediction #{inference_engine.predictions_made:04d}] "
                      f"Command: {prediction['command']:8s} "
                      f"(confidence: {prediction['confidence']:.3f}, "
                      f"rate: {prediction['inference_rate_hz']:.1f} Hz)")
                
                # Publish to robot commands topic
                producer.send(output_topic, prediction)
                
                # Write to log file
                if log_f:
                    log_f.write(json.dumps(prediction) + "\n")
                    log_f.flush()
    
    except KeyboardInterrupt:
        print("\n\nShutdown by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\nShutting down...")
        print(f"Total predictions made: {inference_engine.predictions_made}")
        
        consumer.close()
        producer.flush()
        producer.close()
        if log_f:
            log_f.close()
        
        print("AI Consumer stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI-powered EEG consumer with real-time robot command inference"
    )
    parser.add_argument(
        "--kafka-servers", default="localhost:9092",
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--input-topic", default="raw-eeg",
        help="Kafka topic to consume EEG samples from"
    )
    parser.add_argument(
        "--output-topic", default="robot-commands",
        help="Kafka topic to produce robot commands to"
    )
    parser.add_argument(
        "--group-id", default="ai-eeg-consumer",
        help="Kafka consumer group ID"
    )
    parser.add_argument(
        "--model-path", default=None,
        help="Path to trained model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--n-channels", type=int, default=32,
        help="Number of EEG channels (8, 14, 32, etc.)"
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Device to run model on"
    )
    parser.add_argument(
        "--log-file", default=None,
        help="Path to log predictions to (JSON lines)"
    )
    
    args = parser.parse_args()
    
    main(
        kafka_servers=args.kafka_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        group_id=args.group_id,
        model_path=args.model_path,
        n_channels=args.n_channels,
        device=args.device,
        log_file=args.log_file
    )
