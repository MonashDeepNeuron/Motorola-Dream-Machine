#!/bin/bash

# Quick Demo: EEG → KUKA Pipeline Test
# This script demonstrates the complete pipeline with sample data

set -e

echo "🎬 EEG → KUKA Pipeline Demo"
echo "=========================="
echo "This demo will:"
echo "1. Start Kafka infrastructure"
echo "2. Stream sample EEG data"
echo "3. Show KUKA arm responding to EEG signals"
echo ""

# Check if Kafka is running
if ! docker ps | grep -q kafka; then
    echo "🚀 Starting Kafka infrastructure..."
    cd config
    docker compose up -d
    cd ..
    
    echo "⏳ Waiting for Kafka to start (30 seconds)..."
    sleep 30
    
    # Create topics
    docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic raw-eeg --partitions 1 --replication-factor 1
    docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic eeg-bandpower --partitions 1 --replication-factor 1
else
    echo "✅ Kafka already running"
fi

# Activate virtual environment
echo "🐍 Activating Python environment..."
source venv/bin/activate 2>/dev/null || {
    echo "❌ Virtual environment not found. Run ./setup_emotiv_kuka.sh first"
    exit 1
}

echo ""
echo "🎮 Starting KUKA EEG Controller (mock mode)..."
echo "   This will show how the KUKA arm would respond to EEG signals"
echo "   Press Ctrl+C to stop the demo"
echo ""

# Start KUKA controller in background
python kuka_eeg_controller.py --kafka-server localhost:9092 --mode combined &
KUKA_PID=$!

# Give it time to start
sleep 3

echo "📡 Starting EEG data stream from sample file..."
echo "   Motor imagery events will control the KUKA arm"
echo ""

# Stream sample data
python producer/producer.py --edf-file S012R14.edf --bootstrap-servers localhost:9092 --speed 2.0 &
PRODUCER_PID=$!

# Start consumer for real-time analysis
python consumer/consumer.py --topic raw-eeg --bootstrap-servers localhost:9092 --write-json &
CONSUMER_PID=$!

# Let it run for a bit
echo "🎬 Demo running... Watch the KUKA responses to EEG events!"
echo "   T0 = Rest → KUKA stops"
echo "   T1 = Left hand imagery → KUKA moves left"  
echo "   T2 = Right hand imagery → KUKA moves right"
echo "   Alpha/Beta bands → Relaxation/focus control"
echo ""
echo "Press Ctrl+C to stop demo"

# Wait for user interrupt
trap 'echo ""; echo "🛑 Stopping demo..."; kill $KUKA_PID $PRODUCER_PID $CONSUMER_PID 2>/dev/null; exit' INT
wait

echo "✅ Demo complete!"
