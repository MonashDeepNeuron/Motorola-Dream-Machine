#!/bin/bash
# Quick EEG → Robot Demo (No AI model required)

cd "$(dirname "$0")"
source venv/bin/activate

echo "╔═══════════════════════════════════════╗"
echo "║   EEG → Robot Pipeline Demo          ║"
echo "╚═══════════════════════════════════════╝"

# Check Kafka
echo "[1/2] Checking Kafka..."
if ! docker exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092 >/dev/null 2>&1; then
    echo "Starting Kafka..."
    cd config && docker compose up -d && cd ..
    sleep 10
fi
echo "✅ Kafka running"

# Create topics
echo "[2/2] Creating topics..."
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic raw-eeg --partitions 1 --replication-factor 1 >/dev/null 2>&1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic band-power --partitions 1 --replication-factor 1 >/dev/null 2>&1
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic robot-commands --partitions 1 --replication-factor 1 >/dev/null 2>&1
echo "✅ Topics ready"

# Find data
if [ -f "../past streamed files/EmotivTest_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00.edf" ]; then
    EDF="../past streamed files/Emotiv Test_mindflux_FLEX2_410914_2025.07.12T14.55.56+10.00.edf"
elif [ -f "S012R14.edf" ]; then
    EDF="S012R14.edf"
else
    echo "❌ No EEG data"; exit 1
fi

mkdir -p logs
echo ""
echo "Starting components..."

# 1. Band power analyzer
python consumer/consumer.py --bootstrap-servers localhost:9092 --topic raw-eeg --write-json > logs/bp.log 2>&1 &
BP_PID=$!
sleep 2
[ ! -z "$BP_PID" ] && kill -0 $BP_PID 2>/dev/null && echo "✅ Band Power ($BP_PID)" || { echo "❌ Band Power failed"; exit 1; }

# 2. Robot controller
python simple_robot_controller.py --bootstrap-servers localhost:9092 --input-topic band-power --output-topic robot-commands > logs/ctrl.log 2>&1 &
CTRL_PID=$!
sleep 2
[ ! -z "$CTRL_PID" ] && kill -0 $CTRL_PID 2>/dev/null && echo "✅ Controller ($CTRL_PID)" || { echo "❌ Controller failed"; kill $BP_PID; exit 1; }

# 3. Mock robot
python integrated_robot_controller.py --robot-type mock --kafka-servers localhost:9092 --input-topic robot-commands > logs/robot.log 2>&1 &
ROBOT_PID=$!
sleep 2
[ ! -z "$ROBOT_PID" ] && kill -0 $ROBOT_PID 2>/dev/null && echo "✅ Robot ($ROBOT_PID)" || { echo "❌ Robot failed"; kill $BP_PID $CTRL_PID; exit 1; }

# 4. EEG producer
python producer/producer.py --edf-file "$EDF" --bootstrap-servers localhost:9092 --speed 5.0 > logs/prod.log 2>&1 &
PROD_PID=$!
sleep 2
[ ! -z "$PROD_PID" ] && kill -0 $PROD_PID 2>/dev/null && echo "✅ Producer ($PROD_PID)" || { echo "❌ Producer failed"; kill $BP_PID $CTRL_PID $ROBOT_PID; exit 1; }

echo ""
echo "🎉 ALL RUNNING!"
echo "Monitor: tail -f logs/ctrl.log logs/robot.log"
echo "Running for 30s..."

sleep 30

echo "Stopping..."
kill $PROD_PID $ROBOT_PID $CTRL_PID $BP_PID 2>/dev/null
sleep 1

echo "✅ Done! Check logs/"
ls -lh logs/*.log consumed_*.json 2>/dev/null
