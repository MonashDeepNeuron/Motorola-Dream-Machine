#!/bin/bash

# EEG → KUKA Setup Script
# For first-time users setting up Emotiv headset → KUKA arm pipeline

set -e

echo "🧠 EEG → KUKA Setup Script"
echo "=========================="
echo "This will set up a two-device pipeline:"
echo "Device 1: Emotiv EEG headset data collection"  
echo "Device 2: Data processing & KUKA arm control"
echo ""

# Detect which device this is
echo "Which device is this?"
echo "1) Device 1 (Emotiv data collection)"
echo "2) Device 2 (Processing & KUKA control)"
read -p "Enter choice (1 or 2): " device_choice

# Common setup for both devices
echo ""
echo "📦 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

if [ "$device_choice" = "1" ]; then
    echo ""
    echo "🎧 Setting up Device 1 (Emotiv Data Collection)"
    echo "=============================================="
    
    # Install LSL
    echo "📡 Installing Lab Streaming Layer (LSL)..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if ! command -v brew &> /dev/null; then
            echo "❌ Homebrew not found. Please install it first:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        brew install labstreaminglayer/tap/lsl
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - pylsl package includes LSL binaries, no system install needed
        echo "✅ Linux detected - LSL will be available via pylsl package"
    else
        echo "⚠️  Please manually install LSL from: https://github.com/sccn/liblsl/releases"
    fi
    
    echo ""
    echo "✅ Device 1 setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Start your Emotiv software (EmotivBCI or EMOTIV Launcher)"
    echo "2. Enable LSL streaming in Emotiv settings"
    echo "3. Get Device 2's IP address"
    echo "4. Run: python producer/live_producer.py --bootstrap-servers <DEVICE_2_IP>:9092"
    echo ""
    echo "Test EEG connection with:"
    echo "  python hardware_test.py --check-streams"

elif [ "$device_choice" = "2" ]; then
    echo ""
    echo "🖥️  Setting up Device 2 (Processing & KUKA Control)"
    echo "================================================="
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Please install Docker Desktop first."
        exit 1
    fi
    
    # Find network IP
    echo "🌐 Finding your network IP address..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        IP=$(ifconfig | grep 'inet ' | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
    else
        IP=$(hostname -I | awk '{print $1}')
    fi
    
    echo "📍 Your network IP: $IP"
    echo "   (Device 1 will connect to this IP)"
    
    # Configure Kafka with network IP
    echo "⚙️  Configuring Kafka for network access..."
    sed -i.bak "s/KAFKA_ADVERTISED_LISTENERS: PLAINTEXT:\/\/[^:]*:9092/KAFKA_ADVERTISED_LISTENERS: PLAINTEXT:\/\/$IP:9092/" config/docker-compose.yml
    
    # Start Kafka
    echo "🚀 Starting Kafka infrastructure..."
    cd config
    docker compose down 2>/dev/null || true
    docker compose up -d
    cd ..
    
    # Wait for Kafka to be ready
    echo "⏳ Waiting for Kafka to start (30 seconds)..."
    sleep 30
    
    # Create topics
    echo "📝 Creating Kafka topics..."
    docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic raw-eeg --partitions 1 --replication-factor 1
    docker exec kafka kafka-topics --bootstrap-server localhost:9092 --create --if-not-exists --topic eeg-bandpower --partitions 1 --replication-factor 1
    
    # Test Kafka
    echo "🧪 Testing Kafka connection..."
    docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
    
    echo ""
    echo "✅ Device 2 setup complete!"
    echo ""
    echo "🔗 Network Configuration:"
    echo "   Your IP address: $IP"
    echo "   Kafka port: 9092"
    echo "   Full address: $IP:9092"
    echo ""
    echo "Next steps:"
    echo "1. Give this IP ($IP) to Device 1"
    echo "2. Start consumer: python consumer/consumer.py --topic raw-eeg --bootstrap-servers localhost:9092 --write-json"
    echo "3. Add KUKA integration to the consumer"
    echo ""
    echo "Test with sample data:"
    echo "  python producer/producer.py --edf-file S012R14.edf --bootstrap-servers localhost:9092"

else
    echo "❌ Invalid choice. Please run again and choose 1 or 2."
    exit 1
fi

echo ""
echo "📚 For detailed instructions, see: FIRST_TIME_USER_GUIDE.md"
echo "🔧 For troubleshooting, run: python hardware_test.py --hardware-guide"
