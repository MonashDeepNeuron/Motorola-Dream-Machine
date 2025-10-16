#!/bin/bash

# Simple: Just generate EEG commands
# You run the robot separately

echo "╔════════════════════════════════════════════════════╗"
echo "║   EEG → Movement Commands Generator               ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EEG_DIR="$SCRIPT_DIR/../eeg_pipeline"
JSONL_FILE="$SCRIPT_DIR/asynchronous_deltas.jsonl"
EDF_FILE="$SCRIPT_DIR/../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00_1.edf"

if [ ! -f "$EDF_FILE" ]; then
    echo "❌ EEG data not found"
    exit 1
fi

echo "📊 Input: Emotiv FLEX2 data"
echo "📁 Output: $JSONL_FILE"
echo ""
echo "⚙️  Settings:"
echo "   - Velocity: 20 mm/s (0.02 m/s)"
echo "   - Safety limits: ±500mm X/Y, 100-800mm Z"
echo "   - Speed: 1.0x (real-time simulation)"
echo ""

# Clear file
echo "" > "$JSONL_FILE"
echo "🧹 Cleared old commands"
echo ""

# Activate venv and run
cd "$EEG_DIR"
source venv/bin/activate

echo "🧠 Processing EEG data..."
echo "═══════════════════════════════════════════════════"
echo ""

python eeg_to_movements.py \
    --edf-file "$EDF_FILE" \
    --output "$JSONL_FILE" \
    --speed 1.0 \
    --velocity-scale 0.02

EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    CMD_COUNT=$(wc -l < "$JSONL_FILE" 2>/dev/null || echo "0")
    
    echo "✅ Success!"
    echo "   Commands generated: $CMD_COUNT"
    echo "   File: $JSONL_FILE"
    echo ""
    echo "🤖 To run robot:"
    echo "   cd $SCRIPT_DIR"
    echo "   python ur_asynchronous.py --robot-ip 172.17.0.2 --json-file asynchronous_deltas.jsonl"
    echo ""
else
    echo "❌ Failed (exit: $EXIT_CODE)"
    exit 1
fi
