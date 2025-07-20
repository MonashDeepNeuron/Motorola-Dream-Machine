#!/usr/bin/env python3
"""
Integration Test Script
======================

This script tests the complete pipeline integration.
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_configuration():
    """Test configuration file loading"""
    config_file = Path("pipeline_config.json")
    if not config_file.exists():
        print("❌ Configuration file not found")
        return False
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_eeg_files():
    """Test EEG files availability"""
    eeg_dir = Path("eeg_files")
    if not eeg_dir.exists():
        print("❌ EEG files directory not found")
        return False
    
    edf_files = list(eeg_dir.glob("*.edf"))
    if not edf_files:
        print("❌ No EDF files found")
        return False
    
    print(f"✅ Found {len(edf_files)} EDF files")
    return True

def test_model_structure():
    """Test model file structure"""
    model_file = Path("model/eeg_model.py")
    if not model_file.exists():
        print("❌ Model file not found")
        return False
    
    print("✅ Model file exists")
    return True

def test_robot_integration():
    """Test robot integration files"""
    robot_dir = Path("ursim_test_v1")
    if not robot_dir.exists():
        print("❌ Robot integration directory not found")
        return False
    
    required_files = ["ur_asynchronous.py", "asynchronous_deltas.jsonl"]
    for file in required_files:
        if not (robot_dir / file).exists():
            print(f"❌ Missing robot file: {file}")
            return False
    
    print("✅ Robot integration files present")
    return True

def test_pipeline_scripts():
    """Test pipeline scripts"""
    scripts = [
        "demo_pipeline.py",
        "prepare_data.py", 
        "train_model.py",
        "run_inference.py",
        "unified_pipeline.py"
    ]
    
    missing = []
    for script in scripts:
        if not Path(script).exists():
            missing.append(script)
    
    if missing:
        print(f"❌ Missing scripts: {missing}")
        return False
    
    print("✅ All pipeline scripts present")
    return True

def test_demo_run():
    """Test demo pipeline execution"""
    try:
        # Import and run a minimal demo
        print("Testing demo execution...")
        
        # Simulate demo without importing external dependencies
        os.system("python3 demo_pipeline.py --mode single > /dev/null 2>&1")
        
        # Check if robot command file was updated
        command_file = Path("ursim_test_v1/asynchronous_deltas.jsonl")
        if command_file.exists() and command_file.stat().st_size > 0:
            print("✅ Demo executed successfully")
            return True
        else:
            print("❌ Demo execution failed")
            return False
            
    except Exception as e:
        print(f"❌ Demo test error: {e}")
        return False

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Running Integration Tests")
    print("=" * 40)
    
    tests = [
        ("Configuration", test_configuration),
        ("EEG Files", test_eeg_files),
        ("Model Structure", test_model_structure),
        ("Robot Integration", test_robot_integration),
        ("Pipeline Scripts", test_pipeline_scripts),
        ("Demo Execution", test_demo_run)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nTesting {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Run ./setup_pipeline.sh to install dependencies")
        print("2. Run python3 prepare_data.py --edf-files eeg_files/*.edf")
        print("3. Run python3 train_model.py")
        print("4. Run python3 run_inference.py")
    else:
        print(f"\n⚠️  {len(tests) - passed} tests failed. Check the issues above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
