#!/bin/bash

# Complete End-to-End Integration Test
# This script runs the full pipeline for demonstration

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   EEG → AI → Robot: Complete Integration Test         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    if [ ! -z "$AI_PID" ]; then
        kill $AI_PID 2>/dev/null || true
        echo "  Stopped AI consumer"
    fi
    
    if [ ! -z "$ROBOT_PID" ]; then
        kill $ROBOT_PID 2>/dev/null || true
        echo "  Stopped robot controller"
    fi
    
    if [ ! -z "$PRODUCER_PID" ]; then
        kill $PRODUCER_PID 2>/dev/null || true
        echo "  Stopped producer"
    fi
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Step 1: Check prerequisites
echo -e "\n${BLUE}[1/7] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker Desktop.${NC}"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker.${NC}"
    exit 1
fi

if [ ! -f "venv/bin/activate" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
else
    source venv/bin/activate
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Step 2: Start Kafka
echo -e "\n${BLUE}[2/7] Starting Kafka infrastructure...${NC}"

cd config
docker compose down >/dev/null 2>&1 || true
docker compose up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start Kafka${NC}"
    exit 1
fi

cd ..
echo -e "${GREEN}✅ Kafka started${NC}"

# Step 3: Wait for Kafka to be ready
echo -e "\n${BLUE}[3/7] Waiting for Kafka to be ready...${NC}"
sleep 15

# Create topics
echo "  Creating topics..."
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic raw-eeg --partitions 1 --replication-factor 1 >/dev/null 2>&1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic robot-commands --partitions 1 --replication-factor 1 >/dev/null 2>&1

echo -e "${GREEN}✅ Kafka ready${NC}"

# Step 4: Check/Create model
echo -e "\n${BLUE}[4/7] Checking AI model...${NC}"

if [ ! -f "../model/checkpoints/best_model.pth" ]; then
    echo -e "${YELLOW}⚠️  No trained model found. Training demo model...${NC}"
    cd ../model
    mkdir -p checkpoints
    
    python train_eeg_model.py \
        --n-elec 64 \
        --n-bands 5 \
        --n-frames 12 \
        --n-classes 5 \
        --epochs 5 \
        --batch-size 16 \
        --device cpu \
        --output-dir checkpoints \
        2>&1 | tail -20
    
    cd ../eeg_pipeline
    echo -e "${GREEN}✅ Model trained (demo mode)${NC}"
else
    echo -e "${GREEN}✅ Found existing model${NC}"
fi

# Step 5: Start AI Consumer
echo -e "\n${BLUE}[5/7] Starting AI consumer...${NC}"

mkdir -p logs

python ai_consumer/ai_consumer.py \
    --kafka-servers localhost:9092 \
    --input-topic raw-eeg \
    --output-topic robot-commands \
    --n-channels 64 \
    --device cpu \
    --log-file logs/predictions_$(date +%Y%m%d_%H%M%S).jsonl \
    > logs/ai_consumer.log 2>&1 &

AI_PID=$!
sleep 3

if ! kill -0 $AI_PID 2>/dev/null; then
    echo -e "${RED}❌ AI consumer failed to start. Check logs/ai_consumer.log${NC}"
    exit 1
fi

echo -e "${GREEN}✅ AI consumer running (PID: $AI_PID)${NC}"

# Step 6: Start Robot Controller
echo -e "\n${BLUE}[6/7] Starting robot controller...${NC}"

python integrated_robot_controller.py \
    --kafka-servers localhost:9092 \
    --input-topic robot-commands \
    --robot-type mock \
    --robot-ip 192.168.1.200 \
    --min-confidence 0.3 \
    > logs/robot_controller.log 2>&1 &

ROBOT_PID=$!
sleep 2

if ! kill -0 $ROBOT_PID 2>/dev/null; then
    echo -e "${RED}❌ Robot controller failed to start. Check logs/robot_controller.log${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Robot controller running (PID: $ROBOT_PID)${NC}"

# Step 7: Start EEG Producer
echo -e "\n${BLUE}[7/7] Starting EEG data stream...${NC}"

if [ ! -f "S012R14.edf" ]; then
    echo -e "${RED}❌ Sample EEG file not found: S012R14.edf${NC}"
    exit 1
fi

python producer/producer.py \
    --edf-file S012R14.edf \
    --bootstrap-servers localhost:9092 \
    --speed 2.0 \
    > logs/producer.log 2>&1 &

PRODUCER_PID=$!
sleep 2

if ! kill -0 $PRODUCER_PID 2>/dev/null; then
    echo -e "${RED}❌ Producer failed to start. Check logs/producer.log${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Producer running (PID: $PRODUCER_PID)${NC}"

# Display status
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              🎉 INTEGRATION TEST RUNNING 🎉            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}Pipeline Status:${NC}"
echo "  📡 EEG Producer:       Running (PID $PRODUCER_PID)"
echo "  🧠 AI Consumer:        Running (PID $AI_PID)"
echo "  🤖 Robot Controller:   Running (PID $ROBOT_PID)"
echo ""
echo -e "${YELLOW}Data Flow:${NC}"
echo "  Emotiv EEG → Kafka (raw-eeg) → AI Model → Kafka (robot-commands) → Robot Arm"
echo ""
echo -e "${GREEN}Monitoring:${NC}"
echo "  • AI predictions:   tail -f logs/predictions_*.jsonl"
echo "  • AI consumer:      tail -f logs/ai_consumer.log"
echo "  • Robot commands:   tail -f logs/robot_controller.log"
echo "  • Producer:         tail -f logs/producer.log"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop the integration test${NC}"
echo ""

# Monitor logs in real-time
echo -e "${BLUE}=== Live Output ===${NC}"
echo ""

# Tail all logs simultaneously
tail -f logs/ai_consumer.log logs/robot_controller.log 2>/dev/null &
TAIL_PID=$!

# Wait for user interrupt or producer to finish
wait $PRODUCER_PID 2>/dev/null

# Cleanup will happen via trap
echo -e "\n${GREEN}Producer finished. Stopping other components...${NC}"
kill $TAIL_PID 2>/dev/null || true

# Wait a bit for final processing
sleep 3

# Show statistics
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  Test Results                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

if [ -f logs/predictions_*.jsonl ]; then
    PRED_COUNT=$(wc -l logs/predictions_*.jsonl 2>/dev/null | head -1 | awk '{print $1}')
    echo -e "${GREEN}✅ AI Predictions made: $PRED_COUNT${NC}"
    
    echo -e "\n${YELLOW}Sample predictions:${NC}"
    head -5 logs/predictions_*.jsonl 2>/dev/null | while read line; do
        CMD=$(echo $line | jq -r '.command' 2>/dev/null || echo "unknown")
        CONF=$(echo $line | jq -r '.confidence' 2>/dev/null || echo "0.0")
        echo "  • Command: $CMD (confidence: $CONF)"
    done
else
    echo -e "${YELLOW}⚠️  No predictions logged${NC}"
fi

echo ""
echo -e "${BLUE}Log files:${NC}"
ls -lh logs/ | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo -e "${GREEN}✅ Integration test complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review logs in logs/ directory"
echo "  2. Connect Emotiv headset for live testing"
echo "  3. Train model with real data"
echo "  4. Connect to real robot arm"
echo ""
echo "See QUICK_START_GUIDE.md for detailed instructions."
