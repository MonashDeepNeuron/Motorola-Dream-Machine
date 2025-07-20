# 🧠🤖 Motorola Dream Machine
## Real-time EEG-to-Robot Control System

**Transform brain signals into robot commands in real-time using deep learning and Emotiv EEG headsets.**

![System Overview](https://img.shields.io/badge/Status-Ready%20for%20Deployment-brightgreen) ![Python](https://img.shields.io/badge/Python-3.7%2B-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What This System Does

The Motorola Dream Machine creates a direct brain-computer interface that:
1. **Captures** EEG brain signals from Emotiv headsets (14 channels, 256Hz)
2. **Processes** signals in real-time with advanced filtering and feature extraction  
3. **Predicts** user intentions using a CNN+GCN+Transformer neural network
4. **Controls** Universal Robot (UR) arms with 7 distinct movement commands
5. **Streams** all data continuously for analysis and improvement

---

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🔍 System Pipeline Explained](#-system-pipeline-explained)
- [📊 Step-by-Step Demonstrations](#-step-by-step-demonstrations)
- [🤖 Model Training & Improvement](#-model-training--improvement)
- [📁 Project Structure](#-project-structure)
- [⚙️ Configuration](#-configuration)
- [🔧 Advanced Usage](#-advanced-usage)
- [🐛 Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/MonashDeepNeuron/Motorola-Dream-Machine.git
cd Motorola-Dream-Machine
chmod +x setup.sh
./setup.sh
```

### 2. Run Demo (No Hardware Required)
```bash
# Activate environment
source venv/bin/activate

# Run 60-second simulation demo
python3 scripts/quick_start.py --demo

# Expected output: EEG simulation → Processing → ML predictions → Robot commands
```

### 3. Connect Real Hardware
```bash
# Install Emotiv SDK (for real EEG)
# Download from: https://www.emotiv.com/developer/

# Run with real EEG headset
python3 src/realtime_system.py
```

---

## 🔍 System Pipeline Explained

### Overview: Brain → Computer → Robot
```
[EEG Headset] → [Signal Processing] → [Feature Extraction] → [ML Model] → [Robot Commands] → [UR Robot]
```

### Pipeline Components:

#### 1. **EEG Signal Acquisition** 📡
- **Input**: Raw brain signals (14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Hardware**: Emotiv EPOC X/Flex headsets
- **Output**: Digital EEG data stream (256 Hz, 16-bit resolution)
- **Fallback**: Realistic brain signal simulation when no headset available

#### 2. **Real-time Signal Processing** ⚡
- **Filters Applied**:
  - Bandpass: 0.5-40 Hz (removes DC drift and high-frequency noise)
  - Notch: 60 Hz (eliminates power line interference)
  - Common Average Reference (CAR) for spatial filtering
- **Windowing**: 4-second sliding windows (1024 samples) with 50% overlap
- **Quality**: Automatic artifact detection and channel quality monitoring

#### 3. **Feature Extraction** 🔬
- **Frequency Bands**:
  - Delta (0.5-4 Hz): Deep sleep, unconscious processes
  - Theta (4-8 Hz): Drowsiness, meditation, memory
  - Alpha (8-13 Hz): Relaxed awareness, closed eyes
  - Beta (13-30 Hz): Active thinking, focus, motor control
  - Gamma (30-40 Hz): Cognitive processing, binding
- **Features Computed**: Power spectral density, band ratios, asymmetry indices
- **Output**: 70-dimensional feature vector per window

#### 4. **Machine Learning Inference** 🧠
- **Architecture**: CNN (spatial) + GCN (connectivity) + Transformer (temporal)
- **Input**: Multi-channel EEG feature sequences
- **Output**: 7 robot command probabilities
- **Commands**: 
  - `move_x_positive` / `move_x_negative` (left/right)
  - `move_y_positive` / `move_y_negative` (forward/backward)  
  - `move_z_positive` / `move_z_negative` (up/down)
  - `stop` (no movement)

#### 5. **Robot Control** 🤖
- **Safety**: Movement limits, collision avoidance, emergency stop
- **Commands**: JSON format with position deltas and confidence scores
- **Logging**: All movements logged to `asynchronous_deltas.jsonl`
- **Simulation**: Full UR robot simulator when no physical robot available

---

## 📊 Step-by-Step Demonstrations

### Demo 1: EEG Signal Processing
See exactly how raw brain signals become clean, analyzed data:

```bash
python3 demos/demo_signal_processing.py
```
**Output**: 
- Raw EEG plots (before/after filtering)
- Frequency analysis charts
- Feature extraction results
- Processing time benchmarks

### Demo 2: Feature Extraction Deep Dive
Understand what the ML model actually "sees":

```bash
python3 demos/demo_feature_extraction.py
```
**Output**:
- 5 frequency band power maps
- Channel connectivity matrices  
- Feature importance rankings
- Real-time feature streaming

### Demo 3: Model Predictions
Watch the neural network make decisions:

```bash
python3 demos/demo_model_inference.py
```
**Output**:
- Prediction confidence scores
- Decision boundary visualization
- Model attention maps
- Command classification results

### Demo 4: Robot Command Generation
See how predictions become robot movements:

```bash
python3 demos/demo_robot_control.py
```
**Output**:
- 3D robot position tracking
- Safety limit enforcement
- Command execution timing
- **Live JSONL streaming** to `output/robot_commands.jsonl`

### Demo 5: Full Pipeline Integration
Run the complete system with detailed logging:

```bash
python3 demos/demo_full_pipeline.py --duration 300 --verbose
```
**Output**:
- End-to-end latency measurements
- Component performance metrics
- **Continuous JSONL streaming** to `output/full_session.jsonl`
- Real-time dashboard (localhost:8080)

---

## 🤖 Model Training & Improvement

### Understanding the Model Architecture

#### Why This Architecture?
```python
# CNN Layer: Spatial patterns across EEG channels
# ↓
# GCN Layer: Brain connectivity relationships  
# ↓
# Transformer: Temporal attention and sequence modeling
# ↓
# Classification: 7 robot commands
```

#### Model Components Explained:

**1. Convolutional Neural Network (CNN)**
- **Purpose**: Detect spatial patterns across EEG electrodes
- **Input**: 14 channels × 1024 time points
- **Filters**: Learn local spatial-temporal features
- **Output**: Spatial feature maps

**2. Graph Convolutional Network (GCN)**  
- **Purpose**: Model brain connectivity and channel relationships
- **Graph**: EEG electrodes as nodes, brain connectivity as edges
- **Learning**: How brain regions communicate during different intentions
- **Output**: Connectivity-aware features

**3. Transformer Network**
- **Purpose**: Long-term temporal dependencies and attention
- **Attention**: Focus on important time periods for classification
- **Memory**: Remember patterns across multiple time windows
- **Output**: Context-aware predictions

### Training Your Own Model

#### Step 1: Data Collection
```bash
# Collect training data with the data collection tool
python3 training/collect_training_data.py --duration 1800 --subject_id "user001"
```
**Generates**:
- `data/training/user001_eeg_raw.h5` (EEG signals)
- `data/training/user001_labels.csv` (movement intentions)
- `data/training/user001_metadata.json` (session info)

#### Step 2: Data Preprocessing
```bash
# Clean and prepare data for training
python3 training/preprocess_data.py --input data/training/ --output data/processed/
```
**Processing Steps**:
1. Artifact removal (eye blinks, muscle activity)
2. Signal normalization and standardization
3. Feature extraction and augmentation
4. Train/validation/test split (70/15/15)

#### Step 3: Model Training
```bash
# Train the neural network
python3 training/train_model.py --config config/training.yaml --epochs 100
```
**Training Process**:
- **Optimizer**: AdamW with learning rate scheduling
- **Loss**: Focal loss (handles class imbalance)
- **Validation**: 5-fold cross-validation
- **Monitoring**: TensorBoard logs, early stopping
- **Output**: Best model saved to `models/best_model.pth`

#### Step 4: Model Evaluation
```bash
# Test model performance
python3 training/evaluate_model.py --model models/best_model.pth --test_data data/processed/test/
```
**Evaluation Metrics**:
- **Accuracy**: Overall classification accuracy
- **F1-Score**: Per-class performance
- **Confusion Matrix**: Classification errors
- **Latency**: Real-time inference speed
- **ROC Curves**: Classification confidence

#### Step 5: Model Deployment
```bash
# Deploy new model to real-time system
python3 training/deploy_model.py --model models/best_model.pth --config config/pipeline.yaml
```

### Advanced Training Options

#### Transfer Learning
```bash
# Fine-tune pre-trained model for new user
python3 training/transfer_learning.py --base_model models/pretrained.pth --user_data data/new_user/
```

#### Data Augmentation
```bash
# Generate more training data
python3 training/augment_data.py --input data/processed/ --output data/augmented/ --factor 5
```

#### Hyperparameter Optimization
```bash
# Find best model parameters
python3 training/optimize_hyperparameters.py --trials 100 --config config/optuna.yaml
```

---

## 📁 Project Structure

### Simplified File Organization

```
Motorola-Dream-Machine/
├── 🚀 QUICK START
│   ├── setup.sh                 # One-command setup
│   ├── scripts/quick_start.py   # Demo launcher
│   └── INSTALLATION_SUCCESS.md  # Setup verification
│
├── 📊 DEMONSTRATIONS  
│   ├── demos/demo_signal_processing.py    # Step 1: EEG processing
│   ├── demos/demo_feature_extraction.py   # Step 2: Feature analysis  
│   ├── demos/demo_model_inference.py      # Step 3: ML predictions
│   ├── demos/demo_robot_control.py        # Step 4: Robot commands
│   └── demos/demo_full_pipeline.py        # Step 5: Complete system
│
├── 🤖 MODEL TRAINING
│   ├── training/collect_training_data.py  # Data collection
│   ├── training/preprocess_data.py        # Data cleaning
│   ├── training/train_model.py            # Model training
│   ├── training/evaluate_model.py         # Performance testing
│   └── training/deploy_model.py           # Model deployment
│
├── ⚙️ CORE SYSTEM
│   ├── src/realtime_system.py             # Main production system
│   ├── src/eeg/                           # EEG processing modules
│   ├── src/model/                         # ML inference modules  
│   ├── src/robot/                         # Robot control modules
│   └── src/utils/                         # Configuration & helpers
│
├── 🔧 CONFIGURATION
│   ├── config/pipeline.yaml               # Main system config
│   ├── config/emotiv.yaml                 # EEG headset settings
│   ├── config/robot.yaml                  # Robot control settings
│   └── config/training.yaml               # Model training config
│
└── 📈 OUTPUT & DATA
    ├── output/robot_commands.jsonl        # Live robot command stream
    ├── output/full_session.jsonl          # Complete session data
    ├── logs/                               # System logs
    ├── data/                               # Training/test data
    └── models/                             # Trained models
```

### Core Python Files Explained

#### Essential Files (You Need These):
- **`src/realtime_system.py`**: The main system - runs everything together
- **`scripts/quick_start.py`**: Easy launcher with demos and testing
- **`setup.sh`**: Automatic installation and environment setup

#### EEG Processing Files:
- **`src/eeg/emotiv_streamer.py`**: Connects to Emotiv headset, streams EEG data
- **`src/eeg/processor.py`**: Filters signals, removes artifacts, prepares data  
- **`src/eeg/features.py`**: Extracts frequency bands and features for ML

#### Machine Learning Files:
- **`src/model/architecture.py`**: CNN+GCN+Transformer neural network definition
- **`src/model/inference.py`**: Real-time ML predictions from EEG features
- **`training/train_model.py`**: Complete model training pipeline

#### Robot Control Files:
- **`src/robot/controller.py`**: UR robot control with safety limits
- **`src/robot/simulator.py`**: Robot simulation when no physical robot

#### Utility Files:
- **`src/utils/config.py`**: YAML configuration management
- **`src/utils/helpers.py`**: Logging, data handling, system checks

---

## ⚙️ Configuration

### Main Configuration Files

#### `config/pipeline.yaml` - Core System Settings
```yaml
# EEG Processing Settings
eeg:
  sampling_rate: 256      # Hz - EEG data collection rate
  window_size: 1024       # samples - 4 seconds at 256 Hz  
  overlap: 0.5           # 50% window overlap
  channels: 14           # Number of EEG channels

# Machine Learning Settings  
model:
  architecture: "cnn_gcn_transformer"
  input_features: 70     # Feature vector size
  num_classes: 7         # Robot commands
  confidence_threshold: 0.7  # Minimum prediction confidence

# Robot Control Settings
robot:
  max_velocity: 0.1      # m/s - Maximum movement speed
  workspace_limits:      # Safe movement boundaries
    x: [-0.5, 0.5]      # meters
    y: [-0.5, 0.5]      
    z: [0.1, 0.6]
```

#### `config/emotiv.yaml` - EEG Headset Settings
```yaml
emotiv:
  headset_type: "EPOC_X"           # Headset model
  client_id: "your_client_id"      # Emotiv app credentials
  client_secret: "your_secret"     # From Emotiv developer portal
  license: "your_license_key"      # Emotiv license
  
  # Channel mapping (standard 10-20 system)
  channels:
    - "AF3"  # Frontal left
    - "F7"   # Frontal left temporal  
    - "F3"   # Frontal left
    # ... (complete 14-channel list)
    
  # Data quality settings
  contact_quality_threshold: 2000  # Electrode impedance limit
  data_quality_threshold: 0.8      # Signal quality minimum
```

#### `config/robot.yaml` - Robot Control Settings
```yaml
robot:
  type: "UR5e"                    # Robot model
  ip_address: "192.168.1.100"     # Robot network address
  port: 30002                     # Control port
  
  # Safety settings
  safety:
    max_acceleration: 0.5         # m/s²
    max_velocity: 0.1            # m/s  
    emergency_stop_distance: 0.05 # meters
    
  # Movement mapping
  commands:
    move_x_positive: [0.05, 0, 0]    # 5cm right
    move_x_negative: [-0.05, 0, 0]   # 5cm left
    move_y_positive: [0, 0.05, 0]    # 5cm forward
    move_y_negative: [0, -0.05, 0]   # 5cm backward
    move_z_positive: [0, 0, 0.05]    # 5cm up
    move_z_negative: [0, 0, -0.05]   # 5cm down
    stop: [0, 0, 0]                  # No movement
```

---

## 🔧 Advanced Usage

### Continuous JSONL Streaming

#### Real-time Robot Command Streaming
The system continuously appends robot commands to JSONL files for analysis:

```bash
# Start system with continuous logging
python3 src/realtime_system.py --output output/session_$(date +%Y%m%d_%H%M%S).jsonl

# Monitor live commands
tail -f output/robot_commands.jsonl
```

**JSONL Format**:
```json
{"timestamp": "2025-07-20T19:18:59.858", "command": "move_y_positive", "confidence": 0.812, "dx": 0.0, "dy": 0.05, "dz": 0.0}
{"timestamp": "2025-07-20T19:19:01.234", "command": "move_x_negative", "confidence": 0.743, "dx": -0.05, "dy": 0.0, "dz": 0.0}
{"timestamp": "2025-07-20T19:19:02.567", "command": "stop", "confidence": 0.891, "dx": 0.0, "dy": 0.0, "dz": 0.0}
```

#### Data Analysis Tools
```bash
# Analyze command patterns
python3 tools/analyze_commands.py --input output/robot_commands.jsonl

# Generate movement heatmaps  
python3 tools/visualize_movements.py --input output/robot_commands.jsonl --output plots/

# Export to other formats
python3 tools/convert_data.py --input output/session.jsonl --format csv
```

### Custom Training Data Collection

#### Interactive Data Collection
```bash
# Collect training data with visual cues
python3 training/interactive_collection.py --subject user001 --duration 1800
```
**Process**:
1. Visual cues show desired movement direction
2. User imagines the movement for 4 seconds
3. EEG signals automatically labeled and saved
4. Data quality monitored in real-time

#### Batch Data Processing
```bash
# Process multiple subjects
python3 training/batch_process.py --input data/raw/ --output data/processed/ --subjects all
```

### Performance Monitoring

#### Real-time Dashboard
```bash
# Start web dashboard
python3 tools/dashboard.py --port 8080

# View at: http://localhost:8080
```
**Dashboard Features**:
- Live EEG signal plots
- Model prediction confidence
- Robot position tracking  
- System performance metrics
- Command success rates

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. "No module named 'src'" Error
```bash
# Ensure you're in the project root directory
cd /path/to/Motorola-Dream-Machine

# Install in development mode
pip install -e .

# Or use the setup script
./setup.sh
```

#### 2. Emotiv Headset Not Connecting
```bash
# Check Emotiv Pro software is running
# Verify credentials in config/emotiv.yaml
# Test connection:
python3 tools/test_emotiv_connection.py
```

#### 3. Model Loading Errors
```bash
# Download pre-trained model
python3 tools/download_pretrained_model.py

# Or train your own
python3 training/train_model.py --quick --epochs 10
```

#### 4. Robot Connection Issues  
```bash
# Test robot simulator
python3 tools/test_robot_simulator.py

# Check network connection to real robot
ping 192.168.1.100  # Replace with your robot IP
```

#### 5. Performance Issues
```bash
# Check system requirements
python3 tools/system_benchmark.py

# Optimize for your hardware
python3 tools/optimize_performance.py
```

### Getting Help

- **Documentation**: Check individual demo files for detailed examples
- **Logs**: System logs in `logs/` directory provide detailed error information  
- **Configuration**: Verify all YAML files have correct settings
- **Hardware**: Use simulation modes when real hardware unavailable

---

## 🎉 Success Indicators

When everything is working correctly, you should see:

✅ **EEG Streaming**: "EEG Connected: True, Streaming: True"  
✅ **Signal Processing**: "Windows Processed: X (Rate: X.XX/s)"  
✅ **Model Inference**: "Last Prediction: [command] (Confidence: X.XXX)"  
✅ **Robot Control**: Commands logged to `asynchronous_deltas.jsonl`  
✅ **Safety Systems**: "Robot Active: True" with emergency stop ready

**The system is ready for brain-controlled robotics! 🧠🤖**

---

*For technical support or contributions, see the project repository.*
