#!/usr/bin/env python3
"""
Installation and Setup Script
=============================

This script helps set up the Motorola Dream Machine EEG-Robot system
by installing dependencies and configuring the environment.
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description, check=True):
    """Run a shell command with error handling"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"💥 {description} failed with error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   This system requires Python 3.7 or higher")
        return False

def install_pip_packages():
    """Install required Python packages"""
    
    # Core dependencies (required)
    core_packages = [
        "numpy>=1.19.0",
        "scipy>=1.6.0", 
        "pyyaml>=5.4.0",
        "pandas>=1.2.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0"
    ]
    
    # Optional dependencies (recommended)
    optional_packages = [
        "torch>=1.8.0",
        "mne>=0.23.0",
        "joblib>=1.0.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0"
    ]
    
    print("📦 Installing core dependencies...")
    core_success = True
    for package in core_packages:
        if not run_command(f"{sys.executable} -m pip install {package}", 
                          f"Installing {package.split('>=')[0]}", check=False):
            core_success = False
    
    if not core_success:
        print("❌ Some core dependencies failed to install")
        return False
    
    print("\n📦 Installing optional dependencies...")
    optional_success = 0
    for package in optional_packages:
        if run_command(f"{sys.executable} -m pip install {package}", 
                      f"Installing {package.split('>=')[0]}", check=False):
            optional_success += 1
    
    print(f"✅ Installed {optional_success}/{len(optional_packages)} optional packages")
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = [
        "config",
        "logs", 
        "data",
        "data/raw",
        "data/processed",
        "models",
        "models/checkpoints",
        "output",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    return True

def setup_emotiv_sdk():
    """Setup Emotiv Cortex SDK (if possible)"""
    print("🧠 Setting up Emotiv Cortex SDK...")
    
    system = platform.system().lower()
    
    if system == "windows":
        print("📋 For Windows:")
        print("   1. Download Cortex SDK from: https://www.emotiv.com/developer/")
        print("   2. Install the EmotivBCI application")
        print("   3. Register for developer account")
        print("   4. Install cortex-python library:")
        print("      pip install git+https://github.com/Emotiv/cortex-python.git")
        
    elif system == "linux":
        print("📋 For Linux:")
        print("   1. Download Cortex SDK from: https://www.emotiv.com/developer/")
        print("   2. Install dependencies: sudo apt-get install libusb-1.0-0-dev")
        print("   3. Install cortex-python library:")
        print("      pip install git+https://github.com/Emotiv/cortex-python.git")
        
    elif system == "darwin":  # macOS
        print("📋 For macOS:")
        print("   1. Download Cortex SDK from: https://www.emotiv.com/developer/")
        print("   2. Install the EmotivBCI application")
        print("   3. Install cortex-python library:")
        print("      pip install git+https://github.com/Emotiv/cortex-python.git")
    
    # Try to install cortex-python
    print("\n🔧 Attempting to install cortex-python...")
    success = run_command(
        f"{sys.executable} -m pip install git+https://github.com/Emotiv/cortex-python.git",
        "Installing cortex-python",
        check=False
    )
    
    if not success:
        print("⚠️  Cortex SDK installation failed - system will use simulation mode")
    
    return True

def create_sample_configs():
    """Create sample configuration files"""
    print("⚙️  Creating sample configuration files...")
    
    # Check if config files already exist
    config_files = [
        "config/pipeline.yaml",
        "config/emotiv.yaml", 
        "config/robot.yaml"
    ]
    
    existing_configs = [f for f in config_files if Path(f).exists()]
    
    if existing_configs:
        print(f"📋 Found existing configs: {', '.join(existing_configs)}")
        response = input("Overwrite existing configurations? (y/N): ")
        if response.lower() != 'y':
            print("Keeping existing configurations")
            return True
    
    # Create basic pipeline config
    pipeline_config = """# EEG Processing Pipeline Configuration
eeg_processing:
  sampling_rate: 256
  window_length: 4.0
  overlap: 0.5
  filters:
    lowpass: 40.0
    highpass: 0.5
    notch: 60.0
  frequency_bands:
    delta: [0.5, 4.0]
    theta: [4.0, 8.0]
    alpha: [8.0, 12.0]
    beta: [12.0, 30.0]
    gamma: [30.0, 40.0]

model:
  architecture:
    n_channels: 14
    n_classes: 7
    sampling_rate: 256
    cnn_filters: [32, 64, 128]
    gcn_hidden: 128
    transformer_dim: 256
    transformer_heads: 8
    transformer_layers: 4
    dropout: 0.3
  save_path: "models/eeg_model.pth"
  scaler_path: "models/feature_scaler.pkl"

inference:
  confidence_threshold: 0.7
  prediction_smoothing: true
  smoothing_window: 5
"""
    
    # Create Emotiv config
    emotiv_config = """# Emotiv Headset Configuration
authentication:
  client_id: "your_client_id_here"
  client_secret: "your_client_secret_here"
  username: "your_username_here"
  password: "your_password_here"

headset:
  device_id: null  # null for auto-detection
  model: "EPOC_X"  # EPOC_X, FLEX, INSIGHT
  channels:
    eeg_channels: ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

streaming:
  eeg_rate: 256
  motion_rate: 32
  streams:
    eeg: true
    motion: false
    dev: false

quality:
  contact_quality:
    threshold: 2000
    check_interval: 5.0
"""
    
    # Create robot config
    robot_config = """# Robot Control Configuration
movement:
  step_size: 0.01  # meters
  speed: 0.1      # m/s
  home_position: [0.0, 0.0, 0.3]
  smoothing:
    enabled: true
    factor: 0.7

safety:
  position_limits:
    - [-0.5, 0.5]  # X axis [min, max] in meters
    - [-0.5, 0.5]  # Y axis [min, max] in meters
    - [0.1, 0.8]   # Z axis [min, max] in meters
  velocity_limits: [0.2, 0.2, 0.2]  # max velocity [x, y, z] m/s
  max_acceleration: 1.0  # m/s²

control:
  confidence_threshold: 0.7

simulation:
  enabled: true
  output_file: "robot_commands.json"
"""
    
    # Write config files
    configs = [
        ("config/pipeline.yaml", pipeline_config),
        ("config/emotiv.yaml", emotiv_config),
        ("config/robot.yaml", robot_config)
    ]
    
    for config_file, content in configs:
        try:
            with open(config_file, 'w') as f:
                f.write(content)
            print(f"  ✅ Created: {config_file}")
        except Exception as e:
            print(f"  ❌ Failed to create {config_file}: {e}")
            return False
    
    print("\n📝 Configuration files created!")
    print("⚠️  Remember to update config/emotiv.yaml with your Emotiv credentials!")
    
    return True

def run_system_test():
    """Run a basic system test"""
    print("🧪 Running basic system test...")
    
    try:
        # Test import of main components
        sys.path.insert(0, 'src')
        
        from src.utils.helpers import check_system_requirements
        
        requirements = check_system_requirements()
        
        print("📋 System Requirements Check:")
        for req, status in requirements.items():
            status_str = "✅ OK" if status else "❌ MISSING"
            print(f"   {req}: {status_str}")
        
        # Count available requirements
        available = sum(requirements.values())
        total = len(requirements)
        
        print(f"\n📊 Requirements: {available}/{total} available")
        
        if available >= total - 2:  # Allow 2 optional dependencies to be missing
            print("✅ System test PASSED - ready to run!")
            return True
        else:
            print("⚠️  System test PARTIAL - some features may not work")
            return True
            
    except Exception as e:
        print(f"❌ System test FAILED: {e}")
        return False

def main():
    print("=" * 60)
    print("MOTOROLA DREAM MACHINE - INSTALLATION SETUP")
    print("=" * 60)
    print("Setting up EEG-to-Robot Control System")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Setup directories
    if not setup_directories():
        print("❌ Directory setup failed")
        return 1
    
    # Install Python packages
    if not install_pip_packages():
        print("❌ Package installation failed")
        return 1
    
    # Setup Emotiv SDK
    setup_emotiv_sdk()
    
    # Create sample configs
    if not create_sample_configs():
        print("❌ Configuration setup failed")
        return 1
    
    # Run system test
    run_system_test()
    
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 60)
    print("Next steps:")
    print("1. Update config/emotiv.yaml with your Emotiv credentials")
    print("2. Run: python scripts/quick_start.py --demo")
    print("3. For full system: python scripts/quick_start.py")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    exit(main())
