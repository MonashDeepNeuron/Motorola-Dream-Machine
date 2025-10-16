# Complete End-to-End Integration Guide

## 🎯 Project Overview

This system enables **brain-controlled robot arm movement** using:
- **Emotiv EEG headset** (Flex 2.0, up to 32 channels)
- **AI model** (EEG2Arm) for intention inference
- **Robot arm** (UR or KUKA) for physical movement

## 🏗️ System Architecture

```
Emotiv Headset → LSL → Kafka → AI Model → Robot Commands → Robot Arm
```

### Complete Pipeline Flow

1. **EEG Acquisition**: Emotiv headset streams brain signals via LSL
2. **Data Streaming**: Producer publishes to Kafka topic `raw-eeg`
3. **AI Inference**: AI consumer processes signals and predicts intentions
4. **Command Generation**: Predictions published to `robot-commands` topic
5. **Robot Control**: Controller executes safe movements

---

## ⚡ Quick Start (5 Minutes)

### Prerequisites
- Python 3.8+
- Docker Desktop
- Emotiv headset (optional for testing)

### Step 1: Setup Environment

```bash
cd eeg_pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install LSL (for live EEG)
# macOS: brew install labstreaminglayer/tap/lsl
# Linux: sudo apt-get install liblsl-dev
# Windows: Download from https://github.com/sccn/liblsl/releases
```

### Step 2: Start Kafka Infrastructure

```bash
cd config
docker compose up -d
cd ..

# Wait 20 seconds for Kafka to start
sleep 20

# Create topics
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --topic raw-eeg --partitions 1 --replication-factor 1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --topic robot-commands --partitions 1 --replication-factor 1
```

### Step 3: Test with Sample Data (No Hardware Required)

```bash
# Terminal 1: Start AI Consumer (with untrained model)
python ai_consumer/ai_consumer.py \
    --kafka-servers localhost:9092 \
    --input-topic raw-eeg \
    --output-topic robot-commands \
    --n-channels 64 \
    --log-file predictions.jsonl

# Terminal 2: Start Robot Controller (mock mode)
python integrated_robot_controller.py \
    --kafka-servers localhost:9092 \
    --input-topic robot-commands \
    --robot-type mock \
    --min-confidence 0.3

# Terminal 3: Stream Sample EEG Data
python producer/producer.py \
    --edf-file S012R14.edf \
    --bootstrap-servers localhost:9092 \
    --speed 1.0
```

You should see:
- ✅ EEG samples streaming
- ✅ AI predictions being made
- ✅ Robot commands being executed (mock)

---

## 🧠 Live Emotiv Integration

### Prerequisites
1. Emotiv headset (Flex, EPOC X, Insight)
2. EmotivPRO or EmotivBCI software
3. LSL enabled in Emotiv software

### Setup Emotiv Streaming

1. **Open EmotivPRO/EmotivBCI**
   - Connect your Emotiv headset
   - Check impedance (should be < 20kΩ)
   - Ensure good signal quality

2. **Enable LSL Streaming**
   - EmotivPRO: Settings → Data Streams → LSL → Enable
   - EmotivBCI: Settings → Enable LSL
   - Sample rate: 256 Hz recommended

3. **Verify Connection**
   ```bash
   python hardware_test.py --check-streams
   ```
   Should show: `Found Emotiv stream: 'EmotivDataStream'`

### Run Live Pipeline

```bash
# Terminal 1: Emotiv Producer
python producer/emotiv_producer.py \
    --bootstrap-servers localhost:9092 \
    --timeout 15.0

# Terminal 2: AI Consumer
python ai_consumer/ai_consumer.py \
    --kafka-servers localhost:9092 \
    --model-path checkpoints/best_model.pth \
    --n-channels 32

# Terminal 3: Robot Controller
python integrated_robot_controller.py \
    --kafka-servers localhost:9092 \
    --robot-type mock \
    --min-confidence 0.6
```

---

## 🤖 Robot Integration

### UR Robot (Universal Robots)

```bash
# Install UR driver
pip install ur-rtde

# Run with real UR robot
python integrated_robot_controller.py \
    --kafka-servers localhost:9092 \
    --robot-type ur \
    --robot-ip 192.168.1.200 \
    --min-confidence 0.7
```

### KUKA Robot

Currently uses mock mode. To integrate real KUKA:
1. Install KUKA Python SDK
2. Update `KUKARobot` class in `integrated_robot_controller.py`
3. Configure robot IP and safety limits

---

## 🧪 Training the AI Model

The system works with untrained model (random predictions) for testing, but needs training for real use.

### Quick Training (Dummy Data)

```bash
cd model

python train_eeg_model.py \
    --n-elec 32 \
    --n-bands 5 \
    --n-frames 12 \
    --n-classes 5 \
    --epochs 10 \
    --batch-size 32 \
    --device cpu \
    --output-dir checkpoints
```

This creates `checkpoints/best_model.pth` that can be loaded by the AI consumer.

### Real Training (With Your Data)

1. **Collect Training Data**
   ```bash
   # Record EEG while performing motor imagery tasks
   python producer/emotiv_producer.py --bootstrap-servers localhost:9092
   
   # Save to file
   python consumer/consumer.py --topic raw-eeg --write-json
   ```

2. **Label Data**
   - 0 = REST
   - 1 = LEFT hand movement imagery
   - 2 = RIGHT hand movement imagery
   - 3 = FORWARD movement imagery
   - 4 = BACKWARD movement imagery

3. **Create Dataset Loader**
   - Replace `DummyEEGDataset` in `train_eeg_model.py`
   - Load your labeled EEG recordings
   - Preprocess to (n_channels, n_bands, n_frames) format

4. **Train**
   ```bash
   python model/train_eeg_model.py --epochs 50 --device cuda
   ```

5. **Use Trained Model**
   ```bash
   python ai_consumer/ai_consumer.py --model-path checkpoints/best_model.pth
   ```

---

## 🔧 Configuration

### Emotiv Channel Configurations

**14-channel (EPOC/Insight):**
```python
['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
```

**32-channel (Flex Extended):**
```python
['AF3', 'AF4', 'F7', 'F3', 'F4', 'F8', 'FC5', 'FC1', 
 'FC2', 'FC6', 'T7', 'C3', 'C4', 'T8', 'CP5', 'CP1',
 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO3',
 'PO4', 'O1', 'O2', 'AF7', 'AF8', 'Fp1', 'Fp2', 'Fz']
```

### Safety Limits

Edit in `integrated_robot_controller.py`:
```python
SafetyLimits(
    max_velocity=0.2,          # m/s
    max_acceleration=0.5,      # m/s²
    min_confidence=0.6,        # 0-1
    command_timeout_ms=2000,   # ms
    workspace_min=[-0.5, -0.5, 0.0, -3.14, -3.14, -3.14],
    workspace_max=[0.5, 0.5, 0.5, 3.14, 3.14, 3.14]
)
```

---

## 📊 Data Flow Example

### EEG Sample (raw-eeg topic)
```json
{
  "device_id": "emotiv_hostname_12345",
  "session_id": "123e4567-e89b-12d3-a456-426614174000",
  "timestamp": "2025-10-08T10:30:45.123Z",
  "seq_number": 1234,
  "sample_rate": 256.0,
  "channels": ["AF3", "AF4", "F7", ...],
  "sample_data": [12.5, -8.3, 15.2, ...],
  "classification_label": null
}
```

### AI Prediction (robot-commands topic)
```json
{
  "command": "LEFT",
  "command_id": 1,
  "confidence": 0.87,
  "probabilities": [0.05, 0.87, 0.03, 0.03, 0.02],
  "inference_rate_hz": 8.5,
  "timestamp": 1696761045.123
}
```

### Robot Execution
```
[Prediction #0042] Command: LEFT     (confidence: 0.870, rate: 8.5 Hz)
⬅️  LEFT (confidence: 0.87)
```

---

## 🐛 Troubleshooting

### Emotiv Not Detected
```bash
# Check LSL streams
python hardware_test.py --check-streams

# Verify Emotiv software is running
# Check LSL is enabled in settings
# Restart Emotiv software
```

### Poor Signal Quality
```bash
# Producer shows quality warnings
⚠️  Signal quality issue: flat (std=0.15μV)

# Solutions:
# 1. Apply saline solution to sensors
# 2. Ensure good skin contact
# 3. Check impedance in Emotiv software (< 20kΩ)
# 4. Clean electrodes
```

### Kafka Connection Failed
```bash
# Check Docker is running
docker ps

# Restart Kafka
cd config
docker compose down
docker compose up -d
```

### AI Model Random Predictions
```
⚠️  WARNING: No trained model found. Using random weights.
```

This is expected! Train the model first:
```bash
python model/train_eeg_model.py --epochs 10
```

---

## 📈 Performance Benchmarks

### Expected Latency
- EEG sampling: 4ms (256 Hz)
- LSL transmission: 10-50ms
- Kafka latency: 5-20ms
- AI inference: 10-50ms (CPU), 2-10ms (GPU)
- Robot response: 50-100ms
- **Total end-to-end: 100-250ms** ✅

### Throughput
- EEG streaming: 256 samples/sec
- AI predictions: 2-10 predictions/sec (depending on window size)
- Robot commands: 1-5 commands/sec (with cooldown)

---

## 🎓 Next Steps

### For Development
1. ✅ Test basic pipeline with sample data
2. ✅ Connect Emotiv headset
3. ✅ Collect training data (motor imagery experiments)
4. ✅ Train AI model with real data
5. ✅ Tune confidence thresholds
6. ✅ Connect to real robot arm

### For Production
1. ⚠️ Implement emergency stop mechanism
2. ⚠️ Add user calibration procedure
3. ⚠️ Implement adaptive thresholds
4. ⚠️ Add data logging and replay
5. ⚠️ Create user interface
6. ⚠️ Add multi-user support

---

## 📚 Documentation Reference

- **Full Review**: `COMPREHENSIVE_REVIEW.md`
- **Setup Guides**: `eeg_pipeline/FIRST_TIME_USER_GUIDE.md`
- **Model Documentation**: `model/model_documentation.pdf`
- **Hardware Guide**: Run `python hardware_test.py --hardware-guide`

---

## 🆘 Support

### Common Issues

**"Import torch could not be resolved"**
```bash
pip install torch
```

**"Import rtde_control could not be resolved"**
```bash
pip install ur-rtde
```

**"No Kafka broker available"**
```bash
# Make sure Docker is running
docker ps | grep kafka

# If not, start it
cd eeg_pipeline/config
docker compose up -d
```

### Getting Help

1. Check `COMPREHENSIVE_REVIEW.md` for detailed analysis
2. Run hardware tests: `python hardware_test.py --hardware-guide`
3. Check Kafka logs: `docker logs kafka`
4. Enable debug logging in Python scripts

---

## 🎉 Success Checklist

- [ ] Kafka running (`docker ps` shows kafka container)
- [ ] Python environment activated (`which python` shows venv)
- [ ] Dependencies installed (`pip list | grep kafka`)
- [ ] Emotiv detected (`python hardware_test.py --check-streams`)
- [ ] Sample data streaming (`python producer/producer.py ...`)
- [ ] AI predictions working (`python ai_consumer/ai_consumer.py ...`)
- [ ] Robot responding (`python integrated_robot_controller.py ...`)
- [ ] Model trained with real data
- [ ] Safety limits configured
- [ ] End-to-end latency < 250ms

---

**Last Updated**: October 8, 2025  
**Version**: 1.0  
**Status**: Fully Integrated & Ready for Testing
