#!/bin/bash

# Complete EEG → UR Robot Test
# Runs robot controller and EEG processor together

echo "╔════════════════════════════════════════════════════╗"
echo "║   EEG-Controlled UR Robot Test                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths
EEG_DIR="$SCRIPT_DIR/../eeg_pipeline"
ROBOT_DIR="$SCRIPT_DIR"
JSONL_FILE="$ROBOT_DIR/asynchronous_deltas.jsonl"
EDF_FILE="$SCRIPT_DIR/../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00_1.edf"

# Check EDF file
if [ ! -f "$EDF_FILE" ]; then
    echo "❌ EEG data file not found: $EDF_FILE"
    exit 1
fi

echo "✅ EEG data: $EDF_FILE"
echo "✅ Output file: $JSONL_FILE"
echo ""

# Clear old commands and create fresh file
echo "" > "$JSONL_FILE"
echo "🧹 Cleared command file"
echo ""

# PIDs for cleanup
ROBOT_PID=""
EEG_PID=""

cleanup() {
    echo ""
    echo "🛑 Stopping..."
    [ ! -z "$EEG_PID" ] && kill $EEG_PID 2>/dev/null && echo "  Stopped EEG processor"
    [ ! -z "$ROBOT_PID" ] && kill $ROBOT_PID 2>/dev/null && echo "  Stopped robot controller"
    echo "✅ Done"
}

trap cleanup EXIT INT TERM

# Start robot controller (waits at end of file for new commands)
echo "🤖 Starting UR robot controller..."
echo "   (Connecting to robot at 127.0.0.1 - ports mapped from Docker)"
echo ""

# Activate venv first
cd "$EEG_DIR"
source venv/bin/activate
cd "$ROBOT_DIR"

python ur_asynchronous.py \
    --robot-ip 127.0.0.1 \
    --json-file asynchronous_deltas.jsonl \
    --acceleration 0.5 \
    --responsiveness 0.5 &

ROBOT_PID=$!
sleep 3

if ! kill -0 $ROBOT_PID 2>/dev/null; then
    echo "❌ Robot controller failed to start"
    echo "   Make sure URSim Docker container is running:"
    echo "   docker ps | grep ursim"
    exit 1
fi

echo "✅ Robot controller running (PID: $ROBOT_PID)"
echo ""

# Wait a bit more for robot to be ready
sleep 2

# Start EEG processor (appends commands to file in real-time)
echo "🧠 Starting EEG processor..."
echo "   (Will generate movement commands from EEG data)"
echo ""
echo "═══════════════════════════════════════════════════"
echo ""

cd "$EEG_DIR"

python eeg_to_movements.py \
    --edf-file "$EDF_FILE" \
    --output "$JSONL_FILE" \
    --speed 1.0 \
    --velocity-scale 0.02 &

EEG_PID=$!

# Monitor while EEG processes
wait $EEG_PID
EEG_EXIT=$?

echo ""
echo "═══════════════════════════════════════════════════"
echo ""

if [ $EEG_EXIT -eq 0 ]; then
    echo "✅ EEG processing complete!"
    
    # Show stats
    CMD_COUNT=$(wc -l < "$JSONL_FILE" 2>/dev/null || echo "0")
    echo "   Commands generated: $CMD_COUNT"
    
    echo ""
    echo "⏳ Waiting 10s for robot to finish executing..."
    sleep 10
    
    echo "✅ Test complete!"
else
    echo "❌ EEG processor failed (exit: $EEG_EXIT)"
    exit 1
fi

echo ""
echo "To run again:"
echo "  cd $ROBOT_DIR"
echo "  bash run_eeg_robot_test.sh"
echo ""
