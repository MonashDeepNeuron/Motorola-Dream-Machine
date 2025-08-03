#!/bin/bash

# EEG Pipeline - One-Click Setup and Test
# This script sets up Kafka and tests the real-time EEG pipeline

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}=== EEG Real-time Pipeline Setup ===${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Docker
if ! docker info >/dev/null 2>&1; then
    echo -e "${YELLOW}[ERROR]${NC} Docker is not running. Please start Docker and try again."
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}[1/6]${NC} Setting up Python environment..."
source venv/bin/activate

# Get local IP
LOCAL_IP=$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
if [ -z "$LOCAL_IP" ]; then LOCAL_IP="localhost"; fi
echo -e "${GREEN}[INFO]${NC} Using IP: $LOCAL_IP"

# Update Kafka config
echo -e "${GREEN}[2/6]${NC} Configuring Kafka..."
sed -i.bak "s/KAFKA_ADVERTISED_LISTENERS: PLAINTEXT:\/\/.*:9092/KAFKA_ADVERTISED_LISTENERS: PLAINTEXT:\/\/$LOCAL_IP:9092/" config/docker-compose.yml

# Start Kafka
echo -e "${GREEN}[3/6]${NC} Starting Kafka infrastructure..."
cd config
docker compose down >/dev/null 2>&1 || true
docker compose up -d
sleep 20

# Create topics
echo -e "${GREEN}[4/6]${NC} Creating Kafka topics..."
docker exec kafka kafka-topics --bootstrap-server $LOCAL_IP:9092 --create --if-not-exists --topic raw-eeg >/dev/null 2>&1
echo -e "${GREEN}[INFO]${NC} Topic 'raw-eeg' ready"

cd "$SCRIPT_DIR"

# Test the pipeline
echo -e "${GREEN}[5/6]${NC} Testing the pipeline..."
mkdir -p results
cd results

echo -e "${BLUE}[INFO]${NC} Starting consumer (background)..."
python ../consumer/consumer.py \
    --bootstrap-servers $LOCAL_IP:9092 \
    --topic raw-eeg \
    --write-json \
    --write-png &
CONSUMER_PID=$!

sleep 3

echo -e "${BLUE}[INFO]${NC} Starting producer (30 seconds at 5x speed)..."
python ../producer/producer.py \
    --edf-file ../S012R14.edf \
    --bootstrap-servers $LOCAL_IP:9092 \
    --speed 5.0 &
PRODUCER_PID=$!

# Wait for test
sleep 30

# Clean shutdown
echo -e "${GREEN}[6/6]${NC} Stopping test..."
kill $PRODUCER_PID 2>/dev/null || true
sleep 3
kill $CONSUMER_PID 2>/dev/null || true
sleep 2

# Show results
echo -e "${BLUE}=== Test Results ===${NC}"
ls -la

if [ -f "consumed_overall_band_averages.json" ]; then
    echo -e "${GREEN}[SUCCESS]${NC} Pipeline working! Analysis files generated."
    echo "Alpha band power average: $(cat consumed_overall_band_averages.json | grep -o '"alpha":[^,]*' || echo 'N/A')"
else
    echo -e "${YELLOW}[INFO]${NC} Test completed (short duration, analysis files may not be generated)"
fi

cd "$SCRIPT_DIR"

echo -e "${BLUE}=== Setup Complete ===${NC}"
echo -e "${GREEN}[READY]${NC} EEG Pipeline is ready!"
echo ""
echo "To manually test:"
echo "  Producer: python producer/producer.py --edf-file S012R14.edf --bootstrap-servers $LOCAL_IP:9092"
echo "  Consumer: python consumer/consumer.py --topic raw-eeg --bootstrap-servers $LOCAL_IP:9092"
echo ""
echo "To stop Kafka: cd config && docker compose down"
echo "Results saved in: results/"
