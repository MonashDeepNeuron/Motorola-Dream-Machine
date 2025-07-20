#!/bin/bash
# Setup script for Unified EEG-to-Robot Pipeline

echo "Setting up Unified EEG-to-Robot Pipeline..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from eeg_pipeline requirements
echo "Installing EEG pipeline dependencies..."
pip install -r eeg_pipeline/requirements.txt

# Install additional dependencies for the unified pipeline
echo "Installing additional dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install development dependencies
pip install jupyter notebook

# Create necessary directories
mkdir -p model/checkpoints
mkdir -p logs
mkdir -p data/processed

echo "Setup complete!"
echo ""
echo "To use the pipeline:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. For training: python unified_pipeline.py --mode train --edf-files eeg_files/*.edf"
echo "3. For inference: python unified_pipeline.py --mode inference --edf-files eeg_files/test.edf"
echo "4. For real-time: python unified_pipeline.py --mode realtime"
