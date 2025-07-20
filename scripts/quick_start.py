#!/usr/bin/env python3
"""
Quick Start Script for Real-time EEG-Robot System
=================================================

This script provides an easy way to launch the complete system with
common configurations and options.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.helpers import print_system_info, check_system_requirements

def check_dependencies():
    """Check if all dependencies are available"""
    print("Checking system requirements...")
    requirements = check_system_requirements()
    
    missing = [req for req, status in requirements.items() if not status]
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\nOptional dependencies:")
        print("- pytorch: For ML model inference (can use mock inference)")
        print("- mne: For advanced EEG processing")
        print("- emotiv_cortex: For real Emotiv headset connection")
        print("\nThe system can run with mock/simulation modes for missing dependencies.")
        return False
    else:
        print("✅ All dependencies are available!")
        return True

def setup_environment():
    """Setup the environment for running"""
    # Create necessary directories
    directories = ['logs', 'data', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Created directory: {directory}")

def run_system_tests():
    """Run basic system tests"""
    print("\n🧪 Running system tests...")
    
    test_commands = [
        {
            'name': 'EEG Processor Test',
            'command': [sys.executable, 'src/eeg/processor.py', '--duration', '5'],
            'optional': False
        },
        {
            'name': 'Feature Extractor Test',
            'command': [sys.executable, 'src/eeg/features.py', '--duration', '2'],
            'optional': False
        },
        {
            'name': 'Robot Controller Test',
            'command': [sys.executable, 'src/robot/controller.py', '--duration', '5'],
            'optional': False
        },
        {
            'name': 'Model Inference Test',
            'command': [sys.executable, 'src/model/inference.py', '--iterations', '10'],
            'optional': True
        }
    ]
    
    test_results = {}
    
    for test in test_commands:
        print(f"\n🔧 Running {test['name']}...")
        try:
            result = subprocess.run(
                test['command'], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"✅ {test['name']}: PASSED")
                test_results[test['name']] = True
            else:
                print(f"❌ {test['name']}: FAILED")
                print(f"Error: {result.stderr}")
                test_results[test['name']] = False
                
                if not test['optional']:
                    print(f"Critical test failed: {test['name']}")
                    return False
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ {test['name']}: TIMEOUT")
            test_results[test['name']] = False
        except Exception as e:
            print(f"💥 {test['name']}: ERROR - {e}")
            test_results[test['name']] = False
    
    # Summary
    passed = sum(test_results.values())
    total = len(test_results)
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    return passed >= (total - 1)  # Allow one optional test to fail

def launch_system(args):
    """Launch the main system"""
    print("\n🚀 Launching Real-time EEG-Robot System...")
    
    # Build command
    cmd = [sys.executable, 'src/realtime_system.py']
    
    if args.config_dir:
        cmd.extend(['--config-dir', args.config_dir])
    
    if args.duration:
        cmd.extend(['--duration', str(args.duration)])
    
    if args.status_interval:
        cmd.extend(['--status-interval', str(args.status_interval)])
    
    if args.save_log:
        cmd.append('--save-log')
    
    try:
        # Run the system
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user")
    except subprocess.CalledProcessError as e:
        print(f"\n💥 System failed with error code {e.returncode}")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Quick Start Script for Real-time EEG-Robot System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quick_start.py --demo                    # Run demo mode
  python scripts/quick_start.py --test                    # Run system tests
  python scripts/quick_start.py --duration 60            # Run for 60 seconds
  python scripts/quick_start.py --config-dir my_config   # Use custom config
        """
    )
    
    parser.add_argument('--demo', action='store_true', 
                       help='Run in demo mode (short duration)')
    parser.add_argument('--test', action='store_true', 
                       help='Run system tests only')
    parser.add_argument('--config-dir', default='config', 
                       help='Configuration directory (default: config)')
    parser.add_argument('--duration', type=int, 
                       help='Run duration in seconds (default: indefinite)')
    parser.add_argument('--status-interval', type=int, default=30,
                       help='Status report interval in seconds (default: 30)')
    parser.add_argument('--save-log', action='store_true',
                       help='Save session log on exit')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip system tests before launch')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency check')
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 60)
    print("MOTOROLA DREAM MACHINE - EEG-to-Robot Control System")
    print("=" * 60)
    print("Real-time brain-computer interface for robot control")
    print("Using EEG signals from Emotiv headsets")
    print("=" * 60)
    
    # System info
    if not args.skip_deps:
        print_system_info()
        if not check_dependencies():
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Exiting...")
                return 1
    
    # Setup environment
    print("\n📦 Setting up environment...")
    setup_environment()
    
    # Demo mode
    if args.demo:
        print("\n🎮 Running in DEMO mode")
        args.duration = 60  # 1 minute demo
        args.save_log = True
        print("Demo will run for 60 seconds with full logging")
    
    # Run tests unless skipped
    if args.test:
        print("\n🧪 Running tests only...")
        success = run_system_tests()
        return 0 if success else 1
    
    if not args.skip_tests and not args.demo:
        print("\n🧪 Running pre-flight tests...")
        if not run_system_tests():
            response = input("\nSome tests failed. Continue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Exiting...")
                return 1
    
    # Launch system
    try:
        launch_system(args)
        print("\n✅ System completed successfully!")
        return 0
    except Exception as e:
        print(f"\n💥 System launch failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
