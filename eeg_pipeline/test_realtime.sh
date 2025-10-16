#!/bin/bash

# Quick Test: Real-Time EEG → Robot (No Kafka!)
# 
# This runs both components:
#   1. EEG processor (generates commands)
#   2. Robot controller (executes commands)

cd "$(dirname "$0")"
source venv/bin/activate

echo "╔═══════════════════════════════════════════════╗"
echo "║  Real-Time EEG → Robot Test (No Kafka!)      ║"
echo "╚═══════════════════════════════════════════════╝"
echo ""

# Find Emotiv data file
if [ -f "../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00_1.edf" ]; then
    EDF_FILE="../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00_1.edf"
    echo "✅ Using Emotiv FLEX2 data"
elif [ -f "S012R14.edf" ]; then
    EDF_FILE="S012R14.edf"
    echo "✅ Using S012R14.edf"
else
    echo "❌ No EEG data files found"
    exit 1
fi

OUTPUT_FILE="robot_commands.json"
echo "📁 Output: $OUTPUT_FILE"
echo ""

# Clean old commands file
rm -f "$OUTPUT_FILE"

echo "Starting components..."
echo ""

# Start robot controller in background
echo "🤖 Starting mock robot controller..."
python file_robot_controller.py \
    --file "$OUTPUT_FILE" \
    --robot-type mock \
    --poll-interval 0.05 \
    > logs/file_robot.log 2>&1 &

ROBOT_PID=$!
sleep 1

if kill -0 $ROBOT_PID 2>/dev/null; then
    echo "   ✅ Robot running (PID: $ROBOT_PID)"
else
    echo "   ❌ Robot failed - check logs/file_robot.log"
    exit 1
fi

# Start EEG processor in foreground
echo ""
echo "📡 Starting EEG → Robot processor..."
echo "   (This will stream commands to $OUTPUT_FILE)"
echo ""
echo "═══════════════════════════════════════════════"
echo ""

python realtime_eeg_to_robot.py \
    --edf-file "$EDF_FILE" \
    --output "$OUTPUT_FILE" \
    --speed 5.0 \
    --window-size 2.0 \
    --movement-scale 30.0

echo ""
echo "═══════════════════════════════════════════════"
echo ""
echo "Stopping robot controller..."
kill $ROBOT_PID 2>/dev/null
sleep 1

echo ""
echo "✅ Test Complete!"
echo ""
echo "Results:"
echo "  📊 Commands: $(wc -l < $OUTPUT_FILE 2>/dev/null || echo '0') generated"
echo "  📝 Robot log: logs/file_robot.log"
echo ""
echo "Review:"
echo "  cat $OUTPUT_FILE | jq '.parameters' | head -20"
echo "  tail -20 logs/file_robot.log"
echo ""
