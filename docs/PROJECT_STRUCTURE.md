# Motorola Dream Machine - Simplified File Structure
# ================================================

This document explains ONLY the essential files you need to understand.
The project has been reorganized to reduce confusion and improve clarity.

## 🚀 ESSENTIAL FILES (START HERE)

### 1. Quick Start & Setup
```
setup.sh                        # ONE-COMMAND SETUP - Run this first!
scripts/quick_start.py           # DEMO LAUNCHER - Test everything works
README.md                       # THIS FILE - Complete documentation
```

### 2. Main System (Production Use)
```
src/realtime_system.py          # MAIN SYSTEM - Runs everything together
config/pipeline.yaml            # CORE SETTINGS - Main configuration
```

### 3. Step-by-Step Learning Demos
```
demos/demo_signal_processing.py    # Step 1: How EEG signals get processed
demos/demo_robot_control.py        # Step 4: How robot commands work
```

## 🧠 EEG PROCESSING (Core Components)

### Essential EEG Files
```
src/eeg/emotiv_streamer.py      # Connects to Emotiv headset
src/eeg/processor.py            # Cleans and filters EEG signals  
src/eeg/features.py             # Extracts features for ML
```

**What Each Does:**
- `emotiv_streamer.py`: Gets raw brain signals from headset (or simulates them)
- `processor.py`: Removes noise, applies filters, makes signals clean
- `features.py`: Converts clean signals into numbers the ML model can understand

## 🤖 MACHINE LEARNING (Model Components)

### Essential ML Files
```
src/model/architecture.py       # Neural network definition (CNN+GCN+Transformer)
src/model/inference.py          # Makes predictions from EEG features
training/train_model.py         # Trains the model (COMPREHENSIVE)
```

**What Each Does:**
- `architecture.py`: The "brain" of the AI - neural network that learns patterns
- `inference.py`: Uses trained model to predict what user wants to do
- `train_model.py`: Teaches the model using example data

## 🦾 ROBOT CONTROL (Movement Components)

### Essential Robot Files
```
src/robot/controller.py         # Controls robot movements safely
src/robot/simulator.py          # Simulates robot when no real robot available
```

**What Each Does:**
- `controller.py`: Sends movement commands to robot with safety limits
- `simulator.py`: Fake robot for testing without real hardware

## 📊 DATA & OUTPUT (Where Results Go)

### Data Flow & Streaming
```
output/robot_commands.jsonl           # LIVE STREAM - All robot commands (NEW FORMAT)
ursim_test_v1/asynchronous_deltas.jsonl  # LEGACY STREAM - Your existing format  
logs/                                 # System logs and debugging info
data/                                 # Training data and processed files
```

**Key Streaming Files:**
- Your `asynchronous_deltas.jsonl` gets CONTINUOUSLY UPDATED when system runs
- New `output/robot_commands.jsonl` has more detailed information
- Both files stream in real-time as commands are executed

## ⚙️ CONFIGURATION (Settings)

### Configuration Files (YAML - Easy to Edit)
```
config/pipeline.yaml            # Main system settings
config/emotiv.yaml             # EEG headset configuration  
config/robot.yaml              # Robot control settings
config/training.yaml           # Model training parameters
```

**What to Edit:**
- `pipeline.yaml`: Change EEG processing, model settings, robot limits
- `emotiv.yaml`: Your Emotiv headset credentials and channel setup
- `robot.yaml`: Robot IP address, movement speeds, safety limits

## 🛠️ UTILITIES (Helper Files)

### Essential Utilities  
```
src/utils/config.py             # Loads YAML configuration files
src/utils/helpers.py            # Logging, data handling, system checks
```

## 🔧 ANALYSIS TOOLS (Understanding Your Data)

### Data Analysis Tools
```
tools/analyze_commands.py       # Analyze your asynchronous_deltas.jsonl file
training/collect_training_data.py   # Collect new training data  
training/evaluate_model.py          # Test model performance
```

**How to Use:**
```bash
# Analyze your existing JSONL data
python3 tools/analyze_commands.py --input ursim_test_v1/asynchronous_deltas.jsonl

# Collect new training data
python3 training/collect_training_data.py --duration 1800 --subject user001

# Train a new model
python3 training/train_model.py --config config/training.yaml --epochs 100
```

## 📁 FOLDERS EXPLAINED

```
├── 🚀 QUICK START
│   ├── setup.sh                 # Automatic setup
│   └── scripts/quick_start.py   # Demo launcher
│
├── 📊 STEP-BY-STEP DEMOS  
│   └── demos/                   # Individual component demonstrations
│
├── 🤖 MODEL TRAINING
│   └── training/                # Complete training pipeline
│
├── ⚙️ CORE SYSTEM
│   ├── src/eeg/                 # EEG processing
│   ├── src/model/               # Machine learning
│   ├── src/robot/               # Robot control
│   └── src/utils/               # Utilities
│
├── 🔧 CONFIGURATION
│   └── config/                  # YAML settings files
│
├── 📈 OUTPUT & DATA
│   ├── output/                  # New JSONL streams
│   ├── ursim_test_v1/          # Your existing JSONL files
│   ├── logs/                    # System logs
│   ├── data/                    # Training data
│   └── models/                  # Trained models
│
└── 🛠️ ANALYSIS TOOLS
    └── tools/                   # Data analysis scripts
```

## 🎯 WHAT EACH COMPONENT DOES

### 1. EEG Signal Path
```
Emotiv Headset → emotiv_streamer.py → processor.py → features.py → ML Model
```

### 2. ML Prediction Path  
```
EEG Features → architecture.py → inference.py → Robot Commands
```

### 3. Robot Control Path
```
ML Predictions → controller.py → Robot Movement → JSONL Logging
```

### 4. Data Streaming Path
```
Robot Commands → asynchronous_deltas.jsonl (your format)
                → robot_commands.jsonl (detailed format)
```

## ✅ QUICK VERIFICATION

To verify everything works:

1. **Setup**: `./setup.sh`
2. **Demo**: `python3 scripts/quick_start.py --demo`
3. **Check Output**: `tail -f ursim_test_v1/asynchronous_deltas.jsonl`

If these work, your system is ready!

## 🧹 FILES REMOVED/SIMPLIFIED

We removed/consolidated confusing files:
- Old scattered EEG processing scripts → Unified in `src/eeg/`
- Multiple robot control files → Unified in `src/robot/`  
- Confusing documentation → Single comprehensive README.md
- Mixed file formats → Standardized YAML configs

## 📖 WHERE TO LEARN MORE

- **Complete Documentation**: README.md (main file)
- **Step-by-Step Learning**: Run demos in `demos/` folder
- **Model Training**: Follow `training/train_model.py` with extensive comments
- **Configuration**: Edit YAML files in `config/` folder
- **Data Analysis**: Use tools in `tools/` folder

**Everything is now organized, documented, and ready for professional use! 🚀**
