#!/bin/bash

# Complete EEG → Robot Test
# Uses your ur_asynchronous.py format (dx, dy, dz, drx, dry, drz)

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║     Complete EEG → Robot Pipeline Test               ║"
echo "║  (Uses ur_asynchronous.py compatible format)         ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate Python environment
source venv/bin/activate

# Find Emotiv data
if [ -f "../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00_1.edf" ]; then
    EDF_FILE="../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00_1.edf"
    echo "✅ Using Emotiv FLEX2 data (32 channels, 256 Hz)"
elif [ -f "S012R14.edf" ]; then
    EDF_FILE="S012R14.edf"
    echo "✅ Using S012R14.edf"
else
    echo "❌ No EEG data found"
    exit 1
fi

OUTPUT_FILE="movements.jsonl"
ROBOT_SCRIPT="../ursim_test_v1/ur_asynchronous.py"

echo "📁 Movement file: $OUTPUT_FILE"
echo ""

# Check if robot script exists
if [ ! -f "$ROBOT_SCRIPT" ]; then
    echo "⚠️  Robot script not found: $ROBOT_SCRIPT"
    echo "   Will generate movements only"
    ROBOT_SCRIPT=""
fi

# Clear old movements
rm -f "$OUTPUT_FILE"
echo "🧹 Cleared old movements"
echo ""

# PIDs for cleanup
EEG_PID=""
ROBOT_PID=""

cleanup() {
    echo ""
    echo "🛑 Stopping processes..."
    [ ! -z "$EEG_PID" ] && kill $EEG_PID 2>/dev/null && echo "  Stopped EEG processor"
    [ ! -z "$ROBOT_PID" ] && kill $ROBOT_PID 2>/dev/null && echo "  Stopped robot"
    echo "✅ Cleanup complete"
}

trap cleanup EXIT INT TERM

# Start robot controller if available
if [ ! -z "$ROBOT_SCRIPT" ]; then
    echo "🤖 Starting UR robot controller..."
    python "$ROBOT_SCRIPT" \
        --robot-ip 127.0.0.1 \
        --json-file "$OUTPUT_FILE" \
        --acceleration 0.5 \
        --responsiveness 0.5 \
        > logs/robot.log 2>&1 &
    
    ROBOT_PID=$!
    sleep 2
    
    if kill -0 $ROBOT_PID 2>/dev/null; then
        echo "   ✅ Robot controller running (PID: $ROBOT_PID)"
        echo "      Logs: logs/robot.log"
    else
        echo "   ❌ Robot failed - check logs/robot.log"
        cat logs/robot.log
        exit 1
    fi
    echo ""
fi

# Start EEG processor
echo "📡 Starting EEG → Movement processor..."
echo "   (Processing $EDF_FILE)"
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""

python eeg_to_movements.py \
    --edf-file "$EDF_FILE" \
    --output "$OUTPUT_FILE" \
    --speed 2.0 \
    --window-size 2.0 \
    --velocity-scale 0.05 &

EEG_PID=$!

# Wait for EEG processor to finish
wait $EEG_PID
EEG_EXIT=$?

echo ""
echo "═══════════════════════════════════════════════════════"
echo ""

if [ $EEG_EXIT -eq 0 ]; then
    echo "✅ EEG Processing Complete!"
    echo ""
    
    # Show stats
    COMMAND_COUNT=$(wc -l < "$OUTPUT_FILE" 2>/dev/null || echo "0")
    echo "📊 Statistics:"
    echo "   Commands generated: $COMMAND_COUNT"
    echo "   Output file: $OUTPUT_FILE"
    
    # Show sample commands
    echo ""
    echo "📝 Sample commands:"
    echo "   First:"
    head -1 "$OUTPUT_FILE" | python3 -m json.tool 2>/dev/null || head -1 "$OUTPUT_FILE"
    echo ""
    echo "   Last:"
    tail -1 "$OUTPUT_FILE" | python3 -m json.tool 2>/dev/null || tail -1 "$OUTPUT_FILE"
    
    # Band distribution
    echo ""
    echo "🧠 Dominant bands:"
    grep -o '"dominant_band": "[^"]*"' "$OUTPUT_FILE" | cut -d'"' -f4 | sort | uniq -c | sort -rn
    
    echo ""
    
    if [ ! -z "$ROBOT_PID" ] && kill -0 $ROBOT_PID 2>/dev/null; then
        echo "⏳ Waiting for robot to finish executing commands..."
        echo "   (Robot is reading from $OUTPUT_FILE)"
        echo "   Press Ctrl+C to stop"
        echo ""
        
        # Wait a bit for robot to process
        sleep 5
        
        echo "✅ Test complete!"
    fi
else
    echo "❌ EEG processing failed (exit code: $EEG_EXIT)"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  🎉 Pipeline Test Complete! 🎉"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_FILE" logs/*.log 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "To replay:"
echo "  python ../ursim_test_v1/ur_asynchronous.py --json-file $OUTPUT_FILE"
echo ""
