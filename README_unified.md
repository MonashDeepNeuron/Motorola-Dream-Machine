# Unified EEG-to-Robot Pipeline

This is a complete, single-system pipeline that processes EEG signals from Emotiv headsets and translates them into robot arm commands for the UR simulator. The pipeline eliminates the need for Kafka and connects all components directly.

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Run the setup script to install dependencies
./setup_pipeline.sh

# Activate the environment
source venv/bin/activate
```

### 2. Prepare Your EEG Data

```bash
# Process EDF files into training data
python prepare_data.py --edf-files eeg_files/*.edf --output-dir data/processed
```

### 3. Train the Model

```bash
# Train the EEG-to-Robot model
python train_model.py --data-dir data/processed --epochs 50
```

### 4. Run Inference

```bash
# Process EEG data and generate robot commands
python run_inference.py --mode data --data-dir data/processed

# Or run a real-time simulation
python run_inference.py --mode simulate
```

### 5. Start Robot Simulation

In a separate terminal:

```bash
# Start UR simulator (if you have it set up)
cd ursim_test_v1
python ur_asynchronous.py --robot-ip 127.0.0.1 --json-file asynchronous_deltas.jsonl
```

## 📁 Project Structure

```
Motorola-Dream-Machine/
├── eeg_files/                          # Raw EEG data files
├── eeg_pipeline/                       # Original Kafka-based pipeline components
├── model/                              # EEG-to-Robot neural network model
├── ursim_test_v1/                      # UR robot simulator integration
├── data/                               # Processed training/testing data
├── logs/                               # Training logs and metrics
├── pipeline_config.json                # Configuration file
├── prepare_data.py                     # EEG data preprocessing
├── train_model.py                      # Model training script
├── run_inference.py                    # Inference and robot control
├── unified_pipeline.py                 # Legacy unified script
├── setup_pipeline.sh                   # Environment setup
└── README_unified.md                   # This file
```

## 🧠 Pipeline Overview

### 1. EEG Data Processing
- **Input**: EDF files from Emotiv headset
- **Processing**: 
  - Load EEG signals using MNE
  - Apply frequency domain analysis (FFT-based band power)
  - Extract features in standard frequency bands (delta, theta, alpha, beta, gamma)
  - Create sliding windows for temporal analysis
- **Output**: Feature matrices (channels × frequency_bands × time_windows)

### 2. Machine Learning Model
- **Architecture**: CNN + Graph Neural Network + Transformer
  - **CNN**: Extracts spatial-temporal features from EEG
  - **GCN**: Models electrode connectivity and spatial relationships
  - **Transformer**: Captures long-range temporal dependencies
- **Input**: Multi-channel frequency band power features
- **Output**: Robot control commands (5 classes: rest, left, right, forward, backward)

### 3. Robot Control
- **Command Mapping**:
  - Class 0: Rest (no movement)
  - Class 1: Move right (+X direction)
  - Class 2: Move left (-X direction)
  - Class 3: Move forward (+Y direction)
  - Class 4: Move backward (-Y direction)
- **Output**: JSONL commands to `ursim_test_v1/asynchronous_deltas.jsonl`
- **Integration**: Works with existing UR simulator code

## ⚙️ Configuration

Edit `pipeline_config.json` to customize:

```json
{
  "pipeline_config": {
    "sample_rate": 256,        // EEG sampling rate
    "window_size": 4.0,        // Analysis window size (seconds)
    "step_size": 2.0,          // Window step size (seconds)
    "n_electrodes": 32,        // Number of EEG channels
    "n_frequency_bands": 5,    // Number of frequency bands
    "n_classes": 5,            // Number of robot commands
    "command_rate": 10         // Robot command rate (Hz)
  },
  "frequency_bands": {
    "delta": [0.5, 4],         // Frequency ranges for each band
    "theta": [4, 8],
    "alpha": [8, 12],
    "beta": [12, 30],
    "gamma": [30, 45]
  }
}
```

## 🔄 Usage Modes

### Training Mode

Train the model on your EDF files:

```bash
# Prepare data
python prepare_data.py --edf-files eeg_files/*.edf

# Train model
python train_model.py --epochs 100 --batch-size 32
```

### Inference Mode

Process EEG data and generate robot commands:

```bash
# Process prepared data
python run_inference.py --mode data

# Real-time simulation
python run_inference.py --mode simulate
```

### Real-time Streaming (Future)

For real-time Emotiv headset integration:

```python
# This requires Emotiv SDK integration
python run_inference.py --mode realtime --device emotiv
```

## 📊 Model Architecture Details

### Input Format
- **Shape**: `(batch_size, n_electrodes, n_frequency_bands, time_steps)`
- **Example**: `(32, 32, 5, 10)` for 32 samples, 32 electrodes, 5 frequency bands, 10 time steps

### Architecture Components

1. **3D CNN Layers**:
   - Depth-wise convolutions for spatial-temporal feature extraction
   - Batch normalization and ReLU activations
   - Temporal pooling to reduce sequence length

2. **Graph Convolutional Networks (GCN)**:
   - Models electrode connectivity
   - Captures spatial relationships between brain regions
   - Edge dropout for regularization

3. **Transformer Encoder**:
   - Multi-head self-attention
   - Positional encoding for temporal awareness
   - Causal masking for real-time compatibility

4. **Classification Head**:
   - Fully connected layers
   - Dropout for regularization
   - 5-class output for robot commands

## 🎯 Performance Optimization

### For Better Accuracy:
1. **Data Quality**: Ensure clean EEG recordings with minimal artifacts
2. **Electrode Placement**: Use consistent 10-20 system placement
3. **Training Data**: Collect labeled data for specific mental tasks
4. **Model Tuning**: Adjust hyperparameters in `pipeline_config.json`

### For Real-time Performance:
1. **GPU Acceleration**: Use CUDA-enabled PyTorch
2. **Model Quantization**: Reduce model precision for faster inference
3. **Buffer Management**: Optimize sliding window processing
4. **Command Smoothing**: Apply temporal filtering to robot commands

## 🔧 Troubleshooting

### Common Issues:

**1. Import Errors**
```bash
# Ensure environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r eeg_pipeline/requirements.txt
```

**2. Model Loading Errors**
```bash
# Check if model weights exist
ls -la model/trained_eeg_model.pth

# Train model if missing
python train_model.py
```

**3. Data Processing Errors**
```bash
# Verify EDF files
python -c "import mne; mne.io.read_raw_edf('eeg_files/your_file.edf', preload=True)"

# Check data dimensions
python prepare_data.py --edf-files eeg_files/test.edf
```

**4. Robot Communication Issues**
```bash
# Check if command file is being written
tail -f ursim_test_v1/asynchronous_deltas.jsonl

# Clear command file if needed
> ursim_test_v1/asynchronous_deltas.jsonl
```

## 🚀 Future Enhancements

### Planned Features:
1. **Real-time Emotiv SDK Integration**: Direct streaming from headset
2. **Advanced Signal Processing**: ICA artifact removal, spatial filtering
3. **Online Learning**: Adaptive model updates during use
4. **Multi-modal Integration**: Combine EEG with other sensors
5. **Advanced Robot Control**: 6-DOF movement, force control
6. **Web Interface**: Real-time monitoring and control dashboard

### Integration Points:
- **Emotiv SDK**: For real-time streaming
- **ROS Integration**: For advanced robot control
- **Unity/Gazebo**: For enhanced simulation
- **TensorBoard**: For training visualization

## 📋 Requirements

### Hardware:
- Emotiv EEG headset (EPOC X, FLEX, etc.)
- Computer with 8GB+ RAM
- GPU recommended for training

### Software:
- Python 3.8+
- PyTorch 1.9+
- MNE Python 1.0+
- NumPy, SciPy, Matplotlib

### Optional:
- UR robot simulator
- CUDA for GPU acceleration
- Jupyter for data exploration

## 📄 License

See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request with clear description

## 📞 Support

For issues and questions:
1. Check troubleshooting section above
2. Search existing GitHub issues
3. Create new issue with detailed description

---

**Note**: This pipeline combines all the existing components (EEG processing, ML model, robot control) into a unified system that works on a single machine without requiring Kafka distribution.
