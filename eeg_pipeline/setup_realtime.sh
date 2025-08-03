#!/bin/bash

# EEG Pipeline Setup for Real-time Hardware
# This script sets up the pipeline for live EEG hardware integration

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== EEG Pipeline Real-time Setup ===${NC}"

# Check OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo -e "${GREEN}[1/5]${NC} Detected macOS"
    LSL_INSTALL_CMD="brew install labstreaminglayer/tap/lsl"
    LSL_PATH="/opt/homebrew/lib"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "${GREEN}[1/5]${NC} Detected Linux"
    LSL_INSTALL_CMD="sudo apt-get install liblsl-dev"
    LSL_PATH="/usr/local/lib"
else
    echo -e "${YELLOW}[1/5]${NC} Detected Windows/Other - manual LSL installation required"
    LSL_INSTALL_CMD="Download from https://github.com/sccn/liblsl/releases"
    LSL_PATH="C:/LSL/lib"
fi

# Check Docker
echo -e "${GREEN}[2/5]${NC} Checking Docker..."
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}[ERROR]${NC} Docker is not running. Please start Docker and try again."
    exit 1
fi

# Setup Python environment
echo -e "${GREEN}[3/5]${NC} Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install LSL library
echo -e "${GREEN}[4/5]${NC} Installing LSL library..."
echo "To enable real-time EEG hardware, run:"
echo "  $LSL_INSTALL_CMD"
echo ""

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "For macOS, also set environment variable:"
    echo "  export DYLD_LIBRARY_PATH=$LSL_PATH"
    echo "  (Add this to your ~/.zshrc or ~/.bash_profile)"
fi

# Setup Kafka and test file-based pipeline
echo -e "${GREEN}[5/5]${NC} Setting up Kafka and testing..."
LOCAL_IP=$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
if [ -z "$LOCAL_IP" ]; then LOCAL_IP="localhost"; fi

# Update Kafka config
sed -i.bak "s/KAFKA_ADVERTISED_LISTENERS: PLAINTEXT:\/\/.*:9092/KAFKA_ADVERTISED_LISTENERS: PLAINTEXT:\/\/$LOCAL_IP:9092/" config/docker-compose.yml

# Start Kafka
cd config
docker compose down >/dev/null 2>&1 || true
docker compose up -d
sleep 15

# Create topics
docker exec kafka kafka-topics --bootstrap-server $LOCAL_IP:9092 --create --if-not-exists --topic raw-eeg >/dev/null 2>&1

cd ..

echo -e "${BLUE}=== Setup Complete ===${NC}"
echo ""
echo -e "${GREEN}✅ File-based pipeline ready${NC}"
echo "Test with: python producer/producer.py --edf-file S012R14.edf"
echo ""
echo -e "${YELLOW}⚠️  For real-time hardware:${NC}"
echo "1. Install LSL: $LSL_INSTALL_CMD"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "2. Set environment: export DYLD_LIBRARY_PATH=$LSL_PATH"
fi
echo "3. Start EEG software with LSL streaming enabled"
echo "4. Test connection: python hardware_test.py --check-streams"
echo "5. Start live producer: python producer/live_producer.py"
echo ""
echo -e "${GREEN}📖 See REALTIME_GUIDE.md for detailed hardware integration${NC}"
echo ""
echo "Current Kafka server: $LOCAL_IP:9092"
