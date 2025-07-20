# 🧠 EEG-to-Robot Control System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()

A complete brain-computer interface system that translates EEG signals from Emotiv headsets into real-time robot arm control commands. This project combines advanced signal processing, machine learning, and robotics to create a direct neural control interface.

## 🎯 **Project Overview**

This system processes neural signals in real-time to control a robot arm through thought patterns. The pipeline extracts frequency domain features from EEG data, processes them through a sophisticated neural network (CNN + Graph Neural Network + Transformer), and generates precise robot movement commands.

### **Key Features**
- 🧠 **Real-time EEG processing** from Emotiv headsets
- 🤖 **Direct robot control** via neural signals
- 🔬 **Advanced ML architecture** (CNN + GCN + Transformer)
- 📊 **Frequency domain analysis** with standard neurological bands
- ⚡ **Low-latency processing** for responsive control
- 🎛️ **Configurable parameters** for different use cases
- 📈 **Training pipeline** for custom models

### **Supported Hardware**
- **EEG Headsets**: Emotiv EPOC X, FLEX, Insight
- **Robots**: Universal Robots (UR3, UR5, UR10) via simulation
- **Platforms**: Linux, Windows, macOS

## 🚀 **Quick Start**

### **1. Installation**

### **1. Installation**

#### **Automated Setup (Recommended)**

**Option A: Bash Script (Linux/macOS)**
```bash
# Clone the repository
git clone https://github.com/MonashDeepNeuron/Motorola-Dream-Machine.git
cd Motorola-Dream-Machine

# Run automated setup (creates venv + installs everything)
./setup.sh

# Activate environment
source venv/bin/activate
# or use: ./activate_env.sh

# Quick test
python3 scripts/quick_start.py --demo
```

**Option B: Python Script**
```bash
# Clone the repository
git clone https://github.com/MonashDeepNeuron/Motorola-Dream-Machine.git
cd Motorola-Dream-Machine

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run automated setup
python3 scripts/setup.py

# Quick test
python3 scripts/quick_start.py --demo
```

#### **Manual Installation**
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install numpy scipy pyyaml pandas matplotlib scikit-learn

# Install optional dependencies (recommended)
pip install torch mne joblib seaborn plotly

# Install Emotiv SDK (for real headset)
pip install git+https://github.com/Emotiv/cortex-python.git
```

### **2. Quick Start**

#### **Demo Mode (No Hardware Required)**
```bash
# Activate virtual environment if created
source venv/bin/activate  # Skip if not using venv

# Run 60-second demo with mock data
python3 scripts/quick_start.py --demo
```

#### **Real-time System**
```bash
# Activate virtual environment if created
source venv/bin/activate  # Skip if not using venv

# Start full real-time system
python3 scripts/quick_start.py

# Run for specific duration
python3 scripts/quick_start.py --duration 300  # 5 minutes

# Run with custom config
python3 scripts/quick_start.py --config-dir my_config
```

#### **Individual Components**
```bash
# Test EEG processing
python3 src/eeg/processor.py --duration 10

# Test robot control
python3 src/robot/controller.py --interactive

# Test model inference
python3 src/model/inference.py --iterations 100

# Test Emotiv streaming
python3 src/eeg/emotiv_streamer.py --duration 30
```

### **3. Configuration Setup**

#### **Emotiv Credentials (Required for Real Headset)**
Edit `config/emotiv.yaml`:
```yaml
authentication:
  client_id: "your_emotiv_client_id"
  client_secret: "your_emotiv_client_secret"
  username: "your_emotiv_username"
  password: "your_emotiv_password"
```

**Get Emotiv credentials:**
1. Sign up at [Emotiv Developer Portal](https://www.emotiv.com/developer/)
2. Create new application to get Client ID and Secret
3. Update the config file with your credentials

#### **System Configuration**
The system uses three main config files in `config/`:
- `pipeline.yaml` - EEG processing and model settings
- `emotiv.yaml` - Headset connection and streaming
- `robot.yaml` - Robot control and safety parameters

All configs have sensible defaults and work out-of-the-box for most users.

## 📋 **System Requirements**

### **Hardware**
- **Computer**: 8GB+ RAM, multi-core CPU
- **GPU**: CUDA-compatible GPU recommended for training
- **EEG Headset**: Emotiv device with SDK access
- **Robot**: UR simulator or physical robot

### **Software**
- **Python**: 3.8 or higher
- **Operating System**: Linux (recommended), Windows, macOS
- **Dependencies**: See `requirements.txt`

## 🏗️ **Architecture**

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│   Emotiv    │    │     EEG      │    │   Machine   │    │    Robot     │
│   Headset   │───▶│  Processing  │───▶│  Learning   │───▶│   Control    │
│             │    │              │    │    Model    │    │              │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
      │                     │                   │                   │
      │              • FFT Analysis      • CNN Feature      • 3-axis movement
      │              • Band Power        • Graph Conv       • Real-time commands
      │              • Preprocessing     • Transformer      • JSON output
      │                                  • Classification
```

### **Pipeline Components**

1. **EEG Signal Acquisition**: Real-time streaming from Emotiv headset
2. **Signal Processing**: Frequency domain analysis and feature extraction
3. **Machine Learning**: Advanced neural network for pattern recognition
4. **Robot Control**: Command generation and robot interface

## 📁 **Project Structure**

```
Motorola-Dream-Machine/
├── 📁 src/                          # Main source code
│   ├── 🧠 eeg/                      # EEG processing modules
│   │   ├── emotiv_streamer.py       # Real-time Emotiv interface
│   │   ├── processor.py             # Signal processing
│   │   └── features.py              # Feature extraction
│   ├── 🤖 model/                    # Machine learning model
│   │   ├── eeg_model.py             # Neural network architecture
│   │   └── inference.py             # Real-time inference
│   ├── 🎛️ robot/                    # Robot control
│   │   └── controller.py            # Robot command interface
│   ├── 🔧 utils/                    # Utilities
│   │   └── helpers.py               # Helper functions
│   └── realtime_system.py           # Main unified system
├── 📁 config/                       # Configuration files
│   ├── pipeline.yaml                # Main configuration
│   ├── emotiv.yaml                  # Emotiv-specific settings
│   └── robot.yaml                   # Robot parameters
├── 📁 scripts/                      # Utility scripts
│   ├── setup.py                     # Automated installation
│   └── quick_start.py               # Easy system launcher
├── 📁 data/                         # Data storage (auto-created)
├── 📁 logs/                         # System logs (auto-created)
├── 📁 models/                       # Model storage (auto-created)
├── requirements.txt                 # Python dependencies
└── README.md                        # This comprehensive guide
```

## 🎓 **Training Your Own Model**

### **Data Collection**
```bash
# Activate environment
source venv/bin/activate

# Collect EEG data with the system (coming soon)
python3 src/realtime_system.py --mode collect --duration 300

# Or use existing EDF files from eeg_files/
python3 src/eeg/processor.py --input eeg_files/mindflux_test.md.edf
```

### **Training Process**
The system currently uses a pre-configured model architecture. Custom training features are in development.

**Current Capabilities:**
- Real-time inference with existing model
- Feature extraction from any EEG data
- Simulation mode for development

**Coming Soon:**
- Custom model training pipeline
- Data labeling interface  
- Transfer learning options

## 🎮 **Usage Examples**

### **Basic Operation**
```bash
# Activate environment first (if using venv)
source venv/bin/activate

# 1. Quick demo (no hardware needed)
python3 scripts/quick_start.py --demo

# 2. Real-time with Emotiv headset
python3 scripts/quick_start.py

# 3. Test individual components
python3 src/eeg/emotiv_streamer.py --duration 30
python3 src/robot/controller.py --interactive
```

### **Advanced Usage**
```bash
# Custom configuration
python3 scripts/quick_start.py --config-dir my_custom_config

# Specific duration
python3 scripts/quick_start.py --duration 300 --save-log

# Component testing
python3 src/eeg/processor.py --channels 14 --duration 10
python3 src/model/inference.py --iterations 100
```

## ⚙️ **Configuration Guide**

The system uses YAML configuration files in the `config/` directory:

### **Main Configuration** (`config/pipeline.yaml`)
```yaml
# EEG Processing Settings
eeg_processing:
  sampling_rate: 256        # Hz - standard for most headsets
  window_length: 4.0        # seconds - analysis window
  overlap: 0.5              # 50% window overlap
  
  # Frequency bands for analysis
  frequency_bands:
    delta: [0.5, 4.0]       # Deep sleep/unconscious
    theta: [4.0, 8.0]       # Meditation/drowsiness  
    alpha: [8.0, 12.0]      # Relaxed awareness
    beta: [12.0, 30.0]      # Active concentration
    gamma: [30.0, 40.0]     # High cognitive activity

# Model settings
model:
  architecture:
    n_channels: 14          # Number of EEG channels
    n_classes: 7            # Robot movement commands
    dropout: 0.3            # Prevent overfitting
  confidence_threshold: 0.7 # Minimum confidence for actions
```

### **Emotiv Configuration** (`config/emotiv.yaml`)
```yaml
# Emotiv headset connection
authentication:
  client_id: "your_id"      # Get from Emotiv Developer Portal
  client_secret: "secret"   # Get from Emotiv Developer Portal
  username: "your_email"    # Your Emotiv account
  password: "your_password" # Your Emotiv password

headset:
  model: "EPOC_X"          # EPOC_X, FLEX, or INSIGHT
  channels:
    eeg_channels: ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", 
                   "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

streaming:
  eeg_rate: 256            # Sampling rate in Hz
  streams:
    eeg: true              # Enable EEG data
    motion: false          # Disable motion sensors
```

### **Robot Configuration** (`config/robot.yaml`)
```yaml
# Robot movement settings
movement:
  step_size: 0.01          # Movement step in meters
  speed: 0.1               # Movement speed m/s
  home_position: [0.0, 0.0, 0.3]  # Starting position

# Safety limits
safety:
  position_limits:         # [min, max] for each axis
    - [-0.5, 0.5]         # X axis limits (meters)
    - [-0.5, 0.5]         # Y axis limits (meters) 
    - [0.1, 0.8]          # Z axis limits (meters)
  max_acceleration: 1.0    # m/s² safety limit

simulation:
  enabled: true            # Use simulation by default
  output_file: "robot_commands.json"  # Command log file
```

## 🤖 **Robot Command System**

### **Movement Commands**
The system translates EEG patterns into 7 different robot commands:

| Command ID | Mental Pattern | Robot Action | Description |
|-----------|---------------|--------------|-------------|
| 0 | Rest/Baseline | Stop | No movement |
| 1 | Left Hand Motor Imagery | Move Left (-X) | Move along negative X axis |
| 2 | Right Hand Motor Imagery | Move Right (+X) | Move along positive X axis |
| 3 | Forward Intent | Move Forward (+Y) | Move along positive Y axis |
| 4 | Backward Intent | Move Backward (-Y) | Move along negative Y axis |
| 5 | Up Intent | Move Up (+Z) | Move along positive Z axis |
| 6 | Down Intent | Move Down (-Z) | Move along negative Z axis |

### **Command Output Format**
Commands are saved to `robot_commands.json` in this format:
```json
{
  "timestamp": 1653123456.789,
  "command": "move_x_positive",
  "parameters": {
    "confidence": 0.85,
    "original_prediction": 1
  },
  "robot_state": {
    "position": [0.05, 0.0, 0.3],
    "velocity": [0.1, 0.0, 0.0],
    "is_moving": true,
    "safety_status": "safe"
  }
}
```

### **Safety Features**
- **Position Limits**: Prevents robot from moving outside safe bounds
- **Emergency Stop**: Immediate halt on safety violations
- **Confidence Threshold**: Only high-confidence predictions trigger movement
- **Smoothing**: Reduces jittery movements from noisy predictions

## 🔬 **Technical Architecture**

### **System Pipeline**
```
EEG Headset → Signal Processing → Feature Extraction → ML Model → Robot Commands
     ↓              ↓                    ↓              ↓            ↓
  Raw EEG      Filter & Window      Time/Freq        Neural        Movement
  256 Hz       4-second chunks      Features         Network       JSON Output
```

### **Neural Network Architecture**
The system uses a hybrid model combining:

1. **CNN Layers**: Extract spatial patterns from EEG channels
2. **Graph Convolution**: Model relationships between brain regions  
3. **Transformer**: Capture temporal dependencies
4. **Classification**: Output 7 movement commands

**Model Specifications:**
- **Input**: 14 EEG channels × 5 frequency bands × time windows
- **Output**: 7-class probability distribution
- **Latency**: <100ms from EEG input to robot command
- **Accuracy**: Depends on user training and signal quality

### **Real-time Performance**
- **EEG Sampling**: 256 Hz continuous streaming
- **Processing Windows**: 4 seconds with 50% overlap  
- **Feature Extraction**: 50+ features per window
- **Prediction Rate**: ~5 predictions per second
- **Command Rate**: Variable based on confidence threshold

## 🔧 **Troubleshooting & FAQ**

### **Common Issues**

#### **Q: "python: command not found"**
**Linux/Ubuntu:**
```bash
# Option 1: Install python-is-python3
sudo apt install python-is-python3

# Option 2: Use python3 explicitly (recommended)
python3 scripts/setup.py

# Option 3: Use the bash setup script
./setup.sh
```

#### **Q: "cortex" module not found**
```bash
# Install Emotiv SDK
pip install git+https://github.com/Emotiv/cortex-python.git

# Or run in simulation mode (no real headset needed)
python scripts/quick_start.py --demo
```

#### **Q: Virtual environment issues**
```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Check if venv is active (should show venv path)
which python
```

#### **Q: Permission denied for Emotiv headset**
- **Linux**: Add user to dialout group: `sudo usermod -a -G dialout $USER`
- **Windows**: Run as Administrator
- **macOS**: Grant USB permissions in System Preferences

#### **Q: Model not found error**
```bash
# The system works without pre-trained models using random weights
# This is normal for initial setup - you can train your own model
python scripts/train_model.py  # Coming soon
```

#### **Q: Robot commands not working**
- Check `robot_commands.json` file is being created
- Verify robot configuration in `config/robot.yaml`
- System runs in simulation mode by default (no physical robot needed)

#### **Q: Low EEG signal quality**
- Ensure electrodes have good skin contact
- Use conductive gel if available  
- Check for electrical interference (WiFi, phones, etc.)
- Try different headset positions

### **Performance Tips**

**For Better Accuracy:**
- Train the system with your specific brain patterns
- Use consistent mental imagery during operation
- Minimize head movement during sessions
- Ensure good electrode contact quality

**For Lower Latency:**
- Close unnecessary applications
- Use wired headset connection when possible
- Adjust buffer sizes in configuration
- Enable GPU acceleration if available

### **System Requirements Check**
```bash
# Check if your system meets requirements
python src/utils/helpers.py

# Test individual components
python scripts/quick_start.py --test
```

## 🧪 **Development & Contributing**

### **Development Setup**
```bash
# Activate environment
source venv/bin/activate

# Install development dependencies
pip install -r requirements.txt

# Test system components
python3 scripts/quick_start.py --test

# Run individual component tests
python3 src/eeg/processor.py --duration 5
python3 src/robot/controller.py --duration 5
python3 src/model/inference.py --iterations 10
```

### **Code Structure**
- **`src/eeg/`**: EEG processing and streaming
- **`src/model/`**: Machine learning inference  
- **`src/robot/`**: Robot control and safety
- **`src/utils/`**: Helper functions
- **`config/`**: YAML configuration files
- **`scripts/`**: Setup and launch scripts

### **Contributing**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Test changes: `python scripts/quick_start.py --test`
4. Commit: `git commit -m "Add new feature"`
5. Push: `git push origin feature/new-feature`
6. Create Pull Request

### **Running Tests**
```bash
# Activate environment
source venv/bin/activate

# System-wide test
python3 scripts/quick_start.py --test

# Individual component tests
python3 src/utils/helpers.py  # Check dependencies
python3 -c "import src.eeg.processor; print('EEG processor OK')"
python3 -c "import src.robot.controller; print('Robot controller OK')"
```

## 📚 **Additional Resources**

- **Configuration Examples**: See `config/` directory for sample YAML files
- **Component Tests**: Individual module tests in each `src/` subdirectory
- **System Logs**: Automatic logging to `logs/` directory during operation

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Emotiv Inc.** for EEG hardware and SDK
- **Universal Robots** for robot simulation tools
- **MNE Python** for signal processing capabilities
- **PyTorch** for machine learning framework

## 📞 **Support & Community**

- **Issues**: [GitHub Issues](https://github.com/MonashDeepNeuron/Motorola-Dream-Machine/issues) for bug reports
- **Discussions**: [GitHub Discussions](https://github.com/MonashDeepNeuron/Motorola-Dream-Machine/discussions) for questions  
- **System Logs**: Check `logs/` directory for detailed error information
- **Configuration Help**: See sample files in `config/` directory

## 🚀 **Citation**

If you use this work in your research, please cite:

```bibtex
@software{eeg_robot_control_2025,
  title={EEG-to-Robot Control System},
  author={Monash DeepNeuron},
  year={2025},
  url={https://github.com/MonashDeepNeuron/Motorola-Dream-Machine}
}
```

---

**⚡ Ready to control robots with your thoughts? Let's get started!** 🧠🤖
