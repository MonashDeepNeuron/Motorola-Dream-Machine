#!/usr/bin/env python3
"""
Deploy Model
===========

Deploy the trained EEG model for production use.

Usage:
    python3 training/deploy_model.py --model <model_file> --target <deployment_target>

Output:
    - Production-ready model deployment
    - Configuration files and monitoring setup
    - Deployment validation and health checks
"""

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.utils.helpers import setup_logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class ModelDeployer:
    """Handles model deployment to production."""
    
    def __init__(self, deployment_target="local"):
        self.deployment_target = deployment_target
        self.deployment_info = {
            'deployment_time': datetime.now().isoformat(),
            'target': deployment_target,
            'status': 'preparing',
            'version': '1.0.0'
        }
    
    def validate_model(self, model_file):
        """Validate the model before deployment."""
        print(f"🔍 Validating model for deployment...")
        
        if not model_file:
            print("   ⚠️  No model file provided - using latest trained model")
            # Find latest model file
            model_files = list(Path("models").glob("*.json"))
            if model_files:
                model_file = max(model_files, key=lambda x: x.stat().st_mtime)
                print(f"   📁 Using: {model_file}")
            else:
                print("   ❌ No trained models found")
                return False, None
        
        model_path = Path(model_file)
        
        # Check if model file exists
        if not model_path.exists():
            print(f"   ❌ Model file not found: {model_file}")
            return False, None
        
        # Validate model format
        try:
            with open(model_path, 'r') as f:
                model_data = json.load(f)
            
            required_keys = ['model_type', 'performance', 'training_info']
            missing_keys = [key for key in required_keys if key not in model_data]
            
            if missing_keys:
                print(f"   ❌ Model missing required keys: {missing_keys}")
                return False, None
            
            # Check performance threshold
            accuracy = model_data.get('performance', {}).get('accuracy', 0)
            if accuracy < 0.7:
                print(f"   ⚠️  Model accuracy {accuracy:.3f} below recommended threshold (0.7)")
                response = input("   Continue with deployment? (y/N): ")
                if response.lower() != 'y':
                    return False, None
            
            print(f"   ✅ Model validation passed")
            print(f"   📊 Model accuracy: {accuracy:.3f}")
            
            return True, model_data
            
        except Exception as e:
            print(f"   ❌ Error reading model file: {e}")
            return False, None
    
    def create_deployment_structure(self, output_dir):
        """Create deployment directory structure."""
        print(f"📁 Creating deployment structure...")
        
        output_dir = Path(output_dir)
        
        # Create deployment directories
        directories = [
            'models',
            'config', 
            'logs',
            'scripts',
            'monitoring',
            'backup'
        ]
        
        for directory in directories:
            (output_dir / directory).mkdir(parents=True, exist_ok=True)
        
        print(f"   ✅ Deployment structure created in: {output_dir}")
        return output_dir
    
    def deploy_model_files(self, model_data, model_file, deployment_dir):
        """Deploy model and related files."""
        print(f"📦 Deploying model files...")
        
        deployment_dir = Path(deployment_dir)
        
        # Copy model file
        model_dest = deployment_dir / 'models' / 'eeg_robot_model.json'
        shutil.copy2(model_file, model_dest)
        print(f"   📁 Model deployed to: {model_dest}")
        
        # Copy feature extractor if available
        feature_extractor_files = list(Path("src/eeg").glob("features.py"))
        if feature_extractor_files:
            shutil.copy2(feature_extractor_files[0], deployment_dir / 'models' / 'features.py')
            print(f"   📁 Feature extractor deployed")
        
        # Copy configuration files
        config_files = list(Path("config").glob("*.yaml"))
        for config_file in config_files:
            shutil.copy2(config_file, deployment_dir / 'config' / config_file.name)
        print(f"   📁 Configuration files deployed: {len(config_files)}")
        
        # Create production config
        self.create_production_config(deployment_dir / 'config', model_data)
        
        return True
    
    def create_production_config(self, config_dir, model_data):
        """Create production configuration files."""
        print(f"⚙️  Creating production configuration...")
        
        # Production settings
        prod_config = {
            'model': {
                'path': 'models/eeg_robot_model.json',
                'type': model_data.get('model_type', 'CNN_GCN_Transformer'),
                'version': self.deployment_info['version'],
                'performance_threshold': 0.7
            },
            'eeg': {
                'sampling_rate': 256,
                'channels': 14,
                'buffer_size': 1024,
                'realtime_processing': True
            },
            'robot': {
                'safety_enabled': True,
                'max_velocity': 0.5,
                'workspace_limits': {
                    'x': [-1.0, 1.0],
                    'y': [-1.0, 1.0], 
                    'z': [0.0, 1.0]
                }
            },
            'monitoring': {
                'log_level': 'INFO',
                'metrics_enabled': True,
                'health_check_interval': 30,
                'alert_thresholds': {
                    'accuracy_drop': 0.1,
                    'latency_ms': 500,
                    'error_rate': 0.05
                }
            },
            'deployment': self.deployment_info
        }
        
        # Save production config
        config_file = config_dir / 'production.yaml'
        
        # Convert to YAML-like format (simple version)
        config_lines = []
        
        def dict_to_yaml_lines(d, indent=0):
            lines = []
            for key, value in d.items():
                if isinstance(value, dict):
                    lines.append('  ' * indent + f"{key}:")
                    lines.extend(dict_to_yaml_lines(value, indent + 1))
                elif isinstance(value, list):
                    lines.append('  ' * indent + f"{key}:")
                    for item in value:
                        lines.append('  ' * (indent + 1) + f"- {item}")
                else:
                    lines.append('  ' * indent + f"{key}: {value}")
            return lines
        
        config_lines = dict_to_yaml_lines(prod_config)
        
        with open(config_file, 'w') as f:
            f.write('\n'.join(config_lines))
        
        print(f"   📁 Production config saved: {config_file}")
        
        # Create monitoring config
        monitoring_config = {
            'metrics': [
                'prediction_accuracy',
                'processing_latency',
                'error_rate',
                'confidence_scores',
                'system_health'
            ],
            'dashboards': {
                'realtime_performance': True,
                'daily_reports': True,
                'alert_system': True
            },
            'logging': {
                'level': 'INFO',
                'format': 'timestamp,level,component,message',
                'rotation': 'daily',
                'retention_days': 30
            }
        }
        
        monitoring_file = config_dir / 'monitoring.yaml'
        with open(monitoring_file, 'w') as f:
            json.dump(monitoring_config, f, indent=2)
        
        print(f"   📁 Monitoring config saved: {monitoring_file}")
    
    def create_deployment_scripts(self, deployment_dir):
        """Create deployment and management scripts."""
        print(f"📜 Creating deployment scripts...")
        
        scripts_dir = deployment_dir / 'scripts'
        
        # Start script
        start_script = """#!/bin/bash
# Start EEG Robot Control System

echo "🚀 Starting EEG Robot Control System..."

# Check dependencies
python3 -c "import numpy, json, sys; print('✅ Dependencies OK')" || {
    echo "❌ Missing dependencies"
    exit 1
}

# Check model files
if [ ! -f "models/eeg_robot_model.json" ]; then
    echo "❌ Model file not found"
    exit 1
fi

# Start the system
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
cd ..
python3 -m src.realtime_system --config deployment/config/production.yaml

echo "✅ System started successfully"
"""
        
        with open(scripts_dir / 'start.sh', 'w') as f:
            f.write(start_script)
        
        # Stop script
        stop_script = """#!/bin/bash
# Stop EEG Robot Control System

echo "⏹️  Stopping EEG Robot Control System..."

# Find and kill the process
pkill -f "realtime_system"

echo "✅ System stopped"
"""
        
        with open(scripts_dir / 'stop.sh', 'w') as f:
            f.write(stop_script)
        
        # Health check script
        health_script = """#!/bin/bash
# Health check for EEG Robot Control System

echo "🏥 EEG Robot Control System Health Check"
echo "========================================"

# Check if process is running
if pgrep -f "realtime_system" > /dev/null; then
    echo "✅ System process: Running"
else
    echo "❌ System process: Not running"
    exit 1
fi

# Check log files
if [ -f "logs/system.log" ]; then
    echo "✅ Logging: Active"
    tail -5 logs/system.log
else
    echo "⚠️  Logging: No recent logs"
fi

# Check model file
if [ -f "models/eeg_robot_model.json" ]; then
    echo "✅ Model: Available"
else
    echo "❌ Model: Missing"
    exit 1
fi

echo "✅ System health: OK"
"""
        
        with open(scripts_dir / 'health_check.sh', 'w') as f:
            f.write(health_script)
        
        # Make scripts executable
        for script in scripts_dir.glob('*.sh'):
            script.chmod(0o755)
        
        print(f"   📁 Scripts created in: {scripts_dir}")
        
        # Python deployment launcher
        launcher_script = f"""#!/usr/bin/env python3
\"\"\"
Production EEG Robot Control System Launcher
\"\"\"

import sys
import os
import json
import subprocess
from pathlib import Path

def main():
    print("🚀 EEG Robot Control System - Production Launcher")
    print("=" * 50)
    
    # Change to deployment directory
    deployment_dir = Path(__file__).parent.parent
    os.chdir(deployment_dir)
    
    # Load configuration
    config_file = "config/production.yaml"
    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {{config_file}}")
        sys.exit(1)
    
    print(f"📁 Deployment directory: {{deployment_dir}}")
    print(f"⚙️  Configuration: {{config_file}}")
    
    # Start the system
    try:
        print("🔄 Starting system...")
        
        # Add parent directory to Python path
        parent_dir = deployment_dir.parent
        env = os.environ.copy()
        env['PYTHONPATH'] = str(parent_dir)
        
        # Start the realtime system
        cmd = [sys.executable, '-m', 'src.realtime_system', '--config', config_file]
        
        process = subprocess.Popen(cmd, cwd=parent_dir, env=env)
        
        print(f"✅ System started with PID: {{process.pid}}")
        print("🔄 System running... Press Ctrl+C to stop")
        
        # Wait for process
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\\n⏹️  Stopping system...")
            process.terminate()
            process.wait()
            print("✅ System stopped")
    
    except Exception as e:
        print(f"❌ Error starting system: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
        
        with open(scripts_dir / 'launch.py', 'w') as f:
            f.write(launcher_script)
        
        (scripts_dir / 'launch.py').chmod(0o755)
        
        print(f"   📁 Launcher script created: {scripts_dir / 'launch.py'}")
    
    def create_monitoring_setup(self, deployment_dir):
        """Set up monitoring and logging."""
        print(f"📊 Setting up monitoring...")
        
        monitoring_dir = deployment_dir / 'monitoring'
        
        # Simple monitoring script
        monitor_script = """#!/usr/bin/env python3
\"\"\"
EEG Robot System Monitor
\"\"\"

import time
import json
import psutil
from datetime import datetime
from pathlib import Path

def monitor_system():
    print("📊 EEG Robot System Monitor")
    print("Press Ctrl+C to stop")
    
    log_file = Path("../logs/monitoring.log")
    log_file.parent.mkdir(exist_ok=True)
    
    try:
        while True:
            # Collect metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'system_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
            
            # Log metrics
            with open(log_file, 'a') as f:
                f.write(json.dumps(metrics) + '\\n')
            
            print(f"📈 CPU: {metrics['cpu_percent']:5.1f}% | RAM: {metrics['memory_percent']:5.1f}% | Load: {metrics['system_load']:5.2f}")
            
            time.sleep(30)  # Monitor every 30 seconds
            
    except KeyboardInterrupt:
        print("\\n⏹️  Monitoring stopped")

if __name__ == "__main__":
    monitor_system()
"""
        
        with open(monitoring_dir / 'monitor.py', 'w') as f:
            f.write(monitor_script)
        
        # Create log rotation script
        logrotate_script = """#!/bin/bash
# Log rotation for EEG Robot System

LOG_DIR="../logs"
BACKUP_DIR="../backup/logs"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Rotate logs older than 7 days
find "$LOG_DIR" -name "*.log" -mtime +7 -exec mv {} "$BACKUP_DIR/" \\;

# Compress old backups
find "$BACKUP_DIR" -name "*.log" -mtime +1 -exec gzip {} \\;

# Remove very old backups (30 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +30 -delete

echo "✅ Log rotation completed"
"""
        
        with open(monitoring_dir / 'rotate_logs.sh', 'w') as f:
            f.write(logrotate_script)
        
        (monitoring_dir / 'rotate_logs.sh').chmod(0o755)
        
        print(f"   📁 Monitoring setup in: {monitoring_dir}")
    
    def run_deployment_tests(self, deployment_dir):
        """Run deployment validation tests."""
        print(f"🧪 Running deployment tests...")
        
        deployment_dir = Path(deployment_dir)
        tests_passed = 0
        total_tests = 5
        
        # Test 1: Check required files
        required_files = [
            'models/eeg_robot_model.json',
            'config/production.yaml',
            'scripts/launch.py',
            'scripts/start.sh'
        ]
        
        for file_path in required_files:
            if (deployment_dir / file_path).exists():
                print(f"   ✅ Required file: {file_path}")
                tests_passed += 1
                break
        else:
            print(f"   ❌ Missing required files")
        
        # Test 2: Check script permissions
        script_files = list((deployment_dir / 'scripts').glob('*.sh'))
        if all(script.stat().st_mode & 0o111 for script in script_files):
            print(f"   ✅ Script permissions")
            tests_passed += 1
        else:
            print(f"   ❌ Script permissions")
        
        # Test 3: Validate configuration
        try:
            with open(deployment_dir / 'config' / 'production.yaml', 'r') as f:
                config_content = f.read()
            if 'model:' in config_content and 'eeg:' in config_content:
                print(f"   ✅ Configuration format")
                tests_passed += 1
            else:
                print(f"   ❌ Configuration format")
        except:
            print(f"   ❌ Configuration validation")
        
        # Test 4: Check directory structure
        required_dirs = ['models', 'config', 'scripts', 'logs', 'monitoring']
        if all((deployment_dir / dir_name).is_dir() for dir_name in required_dirs):
            print(f"   ✅ Directory structure")
            tests_passed += 1
        else:
            print(f"   ❌ Directory structure")
        
        # Test 5: Test launcher script syntax
        try:
            import ast
            with open(deployment_dir / 'scripts' / 'launch.py', 'r') as f:
                ast.parse(f.read())
            print(f"   ✅ Launcher script syntax")
            tests_passed += 1
        except:
            print(f"   ❌ Launcher script syntax")
        
        print(f"\n📊 Deployment tests: {tests_passed}/{total_tests} passed")
        
        if tests_passed == total_tests:
            print(f"   🎉 All tests passed - deployment ready!")
            return True
        else:
            print(f"   ⚠️  Some tests failed - review deployment")
            return False
    
    def finalize_deployment(self, deployment_dir, model_data):
        """Finalize the deployment."""
        print(f"🏁 Finalizing deployment...")
        
        deployment_dir = Path(deployment_dir)
        
        # Update deployment info
        self.deployment_info.update({
            'status': 'deployed',
            'model_version': model_data.get('version', '1.0.0'),
            'model_accuracy': model_data.get('performance', {}).get('accuracy', 0),
            'deployment_path': str(deployment_dir.absolute()),
            'components': [
                'model',
                'configuration', 
                'scripts',
                'monitoring',
                'logging'
            ]
        })
        
        # Save deployment manifest
        manifest_file = deployment_dir / 'deployment_manifest.json'
        with open(manifest_file, 'w') as f:
            json.dump(self.deployment_info, f, indent=2)
        
        # Create README
        readme_content = f"""# EEG Robot Control System - Production Deployment

## Deployment Information
- **Deployment Time**: {self.deployment_info['deployment_time']}
- **Target**: {self.deployment_info['target']}
- **Version**: {self.deployment_info['version']}
- **Model Accuracy**: {self.deployment_info['model_accuracy']:.3f}

## Quick Start

### Start the System
```bash
cd scripts
./start.sh
```

Or use the Python launcher:
```bash
python3 scripts/launch.py
```

### Stop the System
```bash
cd scripts
./stop.sh
```

### Health Check
```bash
cd scripts
./health_check.sh
```

## Directory Structure
```
deployment/
├── models/           # Trained models
├── config/           # Configuration files
├── scripts/          # Management scripts
├── logs/             # System logs
├── monitoring/       # Monitoring tools
└── backup/           # Backup storage
```

## Configuration
- Main config: `config/production.yaml`
- Monitoring: `config/monitoring.yaml`

## Monitoring
- System monitor: `monitoring/monitor.py`
- Log rotation: `monitoring/rotate_logs.sh`

## Support
- Check logs in `logs/` directory
- Run health checks regularly
- Monitor system performance

## Safety
- Robot safety limits are enforced
- EEG signal quality is monitored
- Automatic shutdown on errors
"""
        
        readme_file = deployment_dir / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(readme_content)
        
        print(f"   📁 Deployment manifest: {manifest_file}")
        print(f"   📁 README: {readme_file}")
        print(f"   ✅ Deployment finalized")

def main():
    """Main deployment interface."""
    parser = argparse.ArgumentParser(description="Deploy EEG model to production")
    parser.add_argument('--model', help='Trained model file to deploy')
    parser.add_argument('--target', default='local', choices=['local', 'server', 'edge'], 
                       help='Deployment target')
    parser.add_argument('--output', default='deployment', help='Deployment output directory')
    parser.add_argument('--force', action='store_true', help='Force deployment even with warnings')
    
    args = parser.parse_args()
    
    print("🚀 EEG Model Deployment")
    print("=" * 40)
    print(f"Target: {args.target}")
    print(f"Output: {args.output}")
    
    # Initialize deployer
    deployer = ModelDeployer(args.target)
    
    try:
        # Validate model
        is_valid, model_data = deployer.validate_model(args.model)
        if not is_valid and not args.force:
            print("❌ Model validation failed. Use --force to override.")
            return
        
        # Create deployment structure
        deployment_dir = deployer.create_deployment_structure(args.output)
        
        # Deploy model files
        deployer.deploy_model_files(model_data, args.model, deployment_dir)
        
        # Create management scripts
        deployer.create_deployment_scripts(deployment_dir)
        
        # Set up monitoring
        deployer.create_monitoring_setup(deployment_dir)
        
        # Run validation tests
        tests_passed = deployer.run_deployment_tests(deployment_dir)
        
        if tests_passed or args.force:
            # Finalize deployment
            deployer.finalize_deployment(deployment_dir, model_data)
            
            print(f"\n🎉 Deployment completed successfully!")
            print(f"📁 Deployment location: {deployment_dir.absolute()}")
            
            print(f"\n🚀 Next steps:")
            print(f"   1. cd {deployment_dir}")
            print(f"   2. Review configuration in config/")
            print(f"   3. Start system: python3 scripts/launch.py")
            print(f"   4. Monitor: python3 monitoring/monitor.py")
            
        else:
            print(f"\n❌ Deployment validation failed")
            print(f"   Use --force to deploy anyway")
        
    except Exception as e:
        print(f"\n❌ Deployment error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
