#!/bin/bash
# Quick Start Script for EEG-to-Robot Pipeline

echo "🧠 EEG-to-Robot Pipeline Quick Start"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pipeline_config.json" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

echo "📋 Current Setup Status:"
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 is available"
else
    echo "❌ Python 3 not found"
    exit 1
fi

# Check virtual environment
if [ -d "venv" ]; then
    echo "✅ Virtual environment exists"
else
    echo "⚠️  Virtual environment not found - will be created by setup"
fi

# Check EEG files
EDF_COUNT=$(find eeg_files -name "*.edf" 2>/dev/null | wc -l)
if [ $EDF_COUNT -gt 0 ]; then
    echo "✅ Found $EDF_COUNT EEG files"
else
    echo "⚠️  No EEG files found in eeg_files/"
fi

echo ""
echo "🚀 Quick Start Options:"
echo ""
echo "1. 🎮 Demo Mode (No setup required)"
echo "   python3 demo_pipeline.py --mode single"
echo ""
echo "2. 🧪 Integration Test"
echo "   python3 test_integration.py"
echo ""
echo "3. 📦 Full Setup & Training"
echo "   ./setup_pipeline.sh"
echo "   source venv/bin/activate"
echo "   python3 prepare_data.py --edf-files eeg_files/*.edf"
echo "   python3 train_model.py"
echo "   python3 run_inference.py"
echo ""

read -p "Choose option (1/2/3) or press Enter to run demo: " choice

case $choice in
    1|"")
        echo ""
        echo "🎮 Running Demo Mode..."
        python3 demo_pipeline.py --mode single
        echo ""
        echo "Demo completed! Check ursim_test_v1/asynchronous_deltas.jsonl for robot commands."
        ;;
    2)
        echo ""
        echo "🧪 Running Integration Tests..."
        python3 test_integration.py
        ;;
    3)
        echo ""
        echo "📦 Running Full Setup..."
        if [ -f "setup_pipeline.sh" ]; then
            chmod +x setup_pipeline.sh
            ./setup_pipeline.sh
            echo ""
            echo "✅ Setup complete! Now activate the environment and run:"
            echo "   source venv/bin/activate"
            echo "   python3 prepare_data.py --edf-files eeg_files/*.edf"
            echo "   python3 train_model.py"
            echo "   python3 run_inference.py"
        else
            echo "❌ setup_pipeline.sh not found"
        fi
        ;;
    *)
        echo "Invalid choice. Running demo..."
        python3 demo_pipeline.py --mode single
        ;;
esac

echo ""
echo "📚 For more information, see:"
echo "   - README_unified.md (Complete documentation)"
echo "   - PIPELINE_SUMMARY.md (Overview of what was built)"
echo "   - pipeline_config.json (Configuration options)"
