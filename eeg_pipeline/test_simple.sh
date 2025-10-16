#!/bin/bash

# Simplified Integration Test (No AI Model Required)
# Uses basic signal processing instead of deep learning

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Simplified EEG → Robot Test (No AI Model)           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    
    if [ ! -z "$SIMPLE_PID" ]; then
        kill $SIMPLE_PID 2>/dev/null || true
        echo "  Stopped simple controller"
    fi
    
    if [ ! -z "$PRODUCER_PID" ]; then
        kill $PRODUCER_PID 2>/dev/null || true
        echo "  Stopped producer"
    fi
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

trap cleanup EXIT INT TERM

# Check prerequisites
echo -e "\n${BLUE}[1/5] Checking prerequisites...${NC}"

if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker not running${NC}"
    exit 1
fi

if [ ! -f "venv/bin/activate" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

source venv/bin/activate
echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Start Kafka
echo -e "\n${BLUE}[2/5] Starting Kafka...${NC}"
cd config
docker compose down >/dev/null 2>&1 || true
docker compose up -d
cd ..

sleep 10
echo -e "${GREEN}✅ Kafka started${NC}"

# Create topics
echo -e "\n${BLUE}[3/5] Creating topics...${NC}"
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic raw-eeg --partitions 1 --replication-factor 1 >/dev/null 2>&1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic robot-commands --partitions 1 --replication-factor 1 >/dev/null 2>&1
echo -e "${GREEN}✅ Topics ready${NC}"

# Find EEG data file
echo -e "\n${BLUE}[4/5] Finding EEG data...${NC}"

if [ -f "S012R14.edf" ]; then
    EDF_FILE="S012R14.edf"
    echo -e "${GREEN}✅ Using: S012R14.edf${NC}"
elif [ -f "../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00.edf" ]; then
    EDF_FILE="../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00.edf"
    echo -e "${GREEN}✅ Using: Emotiv recorded data${NC}"
else
    echo -e "${RED}❌ No EEG data files found${NC}"
    exit 1
fi

# Start simple controller (no AI model)
echo -e "\n${BLUE}[5/5] Starting simple EEG→Robot pipeline...${NC}"

mkdir -p logs

# Start simple consumer that uses band power instead of AI
python consumer/consumer.py \
    --bootstrap-servers localhost:9092 \
    --topic raw-eeg \
    --write-json \
    > logs/simple_consumer.log 2>&1 &

SIMPLE_PID=$!
sleep 3

if ! kill -0 $SIMPLE_PID 2>/dev/null; then
    echo -e "${RED}❌ Consumer failed. Check logs/simple_consumer.log${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Simple controller running (PID: $SIMPLE_PID)${NC}"

# Start producer
echo -e "\n${BLUE}Streaming EEG data...${NC}"

python producer/producer.py \
    --edf-file "$EDF_FILE" \
    --bootstrap-servers localhost:9092 \
    --speed 5.0 \
    > logs/producer.log 2>&1 &

PRODUCER_PID=$!
sleep 2

if ! kill -0 $PRODUCER_PID 2>/dev/null; then
    echo -e "${RED}❌ Producer failed. Check logs/producer.log${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Producer running (PID: $PRODUCER_PID)${NC}"

echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              🎉 TEST RUNNING 🎉                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}Pipeline:${NC}"
echo "  📡 EEG File → Kafka → Band Power Analysis"
echo ""
echo -e "${YELLOW}Monitor:${NC}"
echo "  tail -f logs/simple_consumer.log"
echo "  tail -f logs/producer.log"
echo ""
echo -e "${BLUE}Running for 30 seconds... (Ctrl+C to stop)${NC}"
echo ""

# Wait for 30 seconds
sleep 30

echo -e "\n${GREEN}Test complete!${NC}"

# Show results
if [ -f "consumed_overall_band_averages.json" ]; then
    echo -e "\n${GREEN}✅ Results generated:${NC}"
    ls -lh consumed_*.* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo -e "${YELLOW}Band power analysis:${NC}"
    cat consumed_overall_band_averages.json | python -m json.tool 2>/dev/null | head -20
else
    echo -e "\n${YELLOW}⚠️  Processing... check logs/${NC}"
fi

echo -e "\n${GREEN}Next: Add AI model for intelligent control${NC}"
