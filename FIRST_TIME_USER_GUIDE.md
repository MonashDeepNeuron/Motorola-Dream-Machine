# EEG → Robot Control: Quick Start Guide

## 🧠🤖 Turn Any EEG Recording into Robot Movements!

This system converts **any EEG file** into robot arm movements using brain wave patterns.

---

## ✨ What You Need

1. **EEG Data File** (`.edf` format)
   - Emotiv headset recordings
   - Clinical EEG recordings
   - Any standard EDF file with EEG channels
   
2. **UR Robot Simulator** (running in Docker)
   ```bash
   docker ps | grep ursim  # Check it's running
   ```
   
3. **Python Environment** (already set up in `eeg_pipeline/venv`)

---

## 🚀 Quick Start: One-Line Command

### Run Complete Test (Robot + EEG Processing)
```bash
cd ursim_test_v1
./run_eeg_robot_test.sh
```

This will:
1. ✅ Connect to robot at 127.0.0.1
2. ✅ Process your EEG data
3. ✅ Generate movement commands in real-time
4. ✅ Watch robot move based on brain waves!

---

## 📊 How It Works

### Brain Wave → Movement Mapping

The system analyzes 5 frequency bands and creates 3D movements:

| Brain Wave | Frequency | Mental State | Robot Movement |
|------------|-----------|--------------|----------------|
| **Delta** δ | 0.5-4 Hz | Deep sleep/rest | ⬅️ Move **Backward** (−X) |
| **Theta** θ | 4-8 Hz | Drowsy/meditation | ⬇️ Move **Down** (−Z) |
| **Alpha** α | 8-13 Hz | Relaxed/calm | ➡️ Move **Forward** (+X) |
| **Beta** β | 13-30 Hz | Focused/alert | ⬆️ Move **Up** (+Z) |
| **Gamma** γ | 30-45 Hz | High concentration | ↗️ Move **Right** (+Y) |

### Multi-Axis Control
- **Primary movement**: Strongest brain wave (>30% power)
- **Secondary movement**: 2nd strongest if >20% power
- **Result**: Diagonal/curved movements when multiple bands are active!

Example:
- 40% Alpha + 25% Beta = Move **forward AND up** (diagonal)
- 35% Gamma + 22% Theta = Move **right AND down**

---

## 🎯 Use Your Own EEG File

### Method 1: Use the Test Script
Edit `run_eeg_robot_test.sh` and change the EDF file path:
```bash
EDF_FILE="/path/to/your/recording.edf"
```

### Method 2: Direct Command
```bash
cd eeg_pipeline
source venv/bin/activate

python eeg_to_movements.py \
    --edf-file "/path/to/your/eeg_data.edf" \
    --output ../ursim_test_v1/asynchronous_deltas.jsonl \
    --speed 2.0 \
    --velocity-scale 0.03
```

### Method 3: Manual (Two Terminals)

**Terminal 1 - Start Robot:**
```bash
cd ursim_test_v1
source ../eeg_pipeline/venv/bin/activate
python ur_asynchronous.py --robot-ip 127.0.0.1 --json-file asynchronous_deltas.jsonl
```

**Terminal 2 - Process EEG:**
```bash
cd eeg_pipeline
source venv/bin/activate
python eeg_to_movements.py \
    --edf-file "YOUR_FILE.edf" \
    --output ../ursim_test_v1/asynchronous_deltas.jsonl
```

---

## ⚙️ Configuration Options

### `eeg_to_movements.py` Parameters

```bash
python eeg_to_movements.py \
    --edf-file <PATH>              # Your EEG file (required)
    --output <PATH>                 # Output JSONL file (default: movements.jsonl)
    --speed <FLOAT>                 # Playback speed multiplier (default: 1.0)
                                    #   1.0 = real-time, 2.0 = 2x faster
    --velocity-scale <FLOAT>        # Movement speed in m/s (default: 0.05)
                                    #   0.02 = 20mm/s (slow & safe)
                                    #   0.05 = 50mm/s (medium)
                                    #   0.10 = 100mm/s (fast)
    --window-size <FLOAT>           # Analysis window in seconds (default: 2.0)
```

### Examples

**Slow & Precise:**
```bash
python eeg_to_movements.py --edf-file data.edf --velocity-scale 0.02 --speed 1.0
```

**Fast Testing:**
```bash
python eeg_to_movements.py --edf-file data.edf --velocity-scale 0.05 --speed 5.0
```

**Maximum Detail:**
```bash
python eeg_to_movements.py --edf-file data.edf --window-size 1.0 --velocity-scale 0.03
```

---

## 🛡️ Safety Features

The system includes automatic safety limits:

- **X-axis**: ±500mm (prevents forward/back crashes)
- **Y-axis**: ±500mm (prevents left/right crashes)  
- **Z-axis**: 100-800mm (prevents hitting table or ceiling)

When approaching limits:
```
⚠️  X limit approaching (0.51m), stopping X motion
```

The robot will **automatically stop** that axis to prevent damage!

---

## 📁 Supported EEG File Formats

### ✅ Works With:
- **Emotiv FLEX2** recordings (32 channels @ 256 Hz)
- **Standard EDF files** with EEG channels
- **Clinical EEG** recordings (PhysioNet, etc.)
- Any file that `mne.io.read_raw_edf()` can load

### Channel Detection:
The system automatically:
- ✅ Detects EEG channels (excludes metadata like timestamps)
- ✅ Works with 14, 32, 64+ channel systems
- ✅ Filters out non-EEG channels automatically

---

## 📈 Monitoring Your Session

### Real-Time Output
The EEG processor shows:
```
Command #10: alpha(45%) + beta(23%) → vel=(0.030, 0.000, 0.015) m/s | pos=(0.15, 0.00, 0.31)m
                ↑              ↑              ↑       ↑       ↑           ↑
           Dominant    Secondary        Forward  Right   Up      Current position
```

### Generated Files
- `asynchronous_deltas.jsonl` - Movement commands (one per line)
- Each line contains: `{"dx": 0.03, "dy": 0, "dz": 0, "drx": 0, "dry": 0, "drz": 0}`

---

## 🎮 Watching the Robot

### VNC Web Interface
Open in your browser:
```
http://localhost:6080/vnc.html
```

You'll see the UR robot simulator executing your brain-controlled movements in real-time!

### What to Expect
- **Alpha-heavy EEG**: Robot moves forward (relaxed state)
- **Beta-heavy EEG**: Robot moves up (focused state)
- **Mixed patterns**: Diagonal/curved movements
- **Changing patterns**: Robot adapts in real-time!

---

## 🔧 Troubleshooting

### Robot Won't Connect
```bash
# Check if URSim is running
docker ps | grep ursim

# Check if ports are accessible
curl -v http://localhost:6080
```

### No Movement
- Robot might be at a safety limit
- Check the EEG processor output for warnings
- Verify file is being updated: `tail -f asynchronous_deltas.jsonl`

### Robot Moves Too Fast/Slow
Adjust `--velocity-scale`:
- Too fast: Use `0.01` or `0.02`
- Too slow: Use `0.05` or `0.10`

---

## 🎓 Understanding the Output

### Example Session
```
2025-10-08 12:10:20 - EEG → Movement converter initialized
2025-10-08 12:10:20 - Velocity scale: 0.03 m/s (30.0 mm/s)
2025-10-08 12:10:20 - Safety limits enabled: X=(-0.5, 0.5), Y=(-0.5, 0.5), Z=(0.1, 0.8)
2025-10-08 12:10:20 - Processing 512 samples per window
2025-10-08 12:10:26 - Cmd #10: alpha(42%) + delta(28%) → vel=(0.025, 0.000, 0.000) m/s | pos=(0.13, 0.00, 0.30)m
```

**What this means:**
- **alpha(42%)**: Alpha waves are dominant at 42% of total power → move forward
- **delta(28%)**: Delta is secondary at 28% → slight backward component
- **vel=(0.025, 0.000, 0.000)**: Net velocity is 25mm/s forward (X-axis)
- **pos=(0.13, 0.00, 0.30)**: Robot is at 130mm forward, 0mm left/right, 300mm height

---

## 🚀 Next Steps

### Add AI Model (Future Enhancement)
Currently using band power analysis. You can train a deep learning model:
```bash
cd model
python train_eeg_model.py --data-dir /path/to/training/data
```

### Try Different Mental States
Record EEG while:
- 🧘 Meditating (high alpha) → smooth forward motion
- 🎯 Concentrating (high beta) → upward motion  
- 😴 Drowsy (high theta) → downward motion
- 🔬 Problem-solving (mixed) → complex 3D paths

### Combine with Real Headset
Replace file playback with live LSL stream:
```bash
python producer/emotiv_producer.py --emotiv-stream
```

---

## 📝 Example Workflows

### 1. Quick Demo with Sample Data
```bash
cd ursim_test_v1
./run_eeg_robot_test.sh
# Uses included Emotiv FLEX2 recording
```

### 2. Your Own Recording
```bash
cd eeg_pipeline
source venv/bin/activate
python eeg_to_movements.py \
    --edf-file ~/my_eeg_data/recording_2025.edf \
    --output ../ursim_test_v1/asynchronous_deltas.jsonl \
    --speed 1.5 \
    --velocity-scale 0.04
```

### 3. High-Speed Playback for Testing
```bash
python eeg_to_movements.py \
    --edf-file data.edf \
    --output commands.jsonl \
    --speed 10.0  # 10x real-time
```

---

## 🎉 Success!

If you see:
- ✅ "Connected to UR at 127.0.0.1"
- ✅ "Cmd #10: alpha(45%) → vel=..."
- ✅ Robot moving in VNC viewer

**Congratulations! Your brain is controlling the robot!** 🧠🤖

---

## 📚 Technical Details

### Signal Processing Pipeline
1. **Load EEG** → MNE library reads EDF file
2. **Window Analysis** → 2-second sliding windows (50% overlap)
3. **Power Spectral Density** → Welch's method computes frequency content
4. **Band Power Extraction** → Integrate PSD in each frequency band
5. **Movement Generation** → Map band ratios to velocities
6. **Safety Check** → Verify limits before applying
7. **Output** → Append to JSONL file

### File Format
Each command line in `asynchronous_deltas.jsonl`:
```json
{
  "dx": 0.030,           // X velocity (m/s)
  "dy": 0.000,           // Y velocity (m/s)
  "dz": 0.000,           // Z velocity (m/s)
  "drx": 0.0,            // X rotation (rad/s)
  "dry": 0.0,            // Y rotation (rad/s)
  "drz": 0.0,            // Z rotation (rad/s)
  "timestamp": "2025-10-08T12:10:20",
  "command_number": 42,
  "dominant_band": "alpha"
}
```

---

**Ready to control robots with your mind? Let's go!** 🚀
