#!/bin/bash
"""
Easy Setup Script for Motorola Dream Machine
===========================================

This script sets up the complete environment including virtual environment,
dependencies, and basic configuration.
"""

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Find python command
find_python() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        # Check if it's Python 3
        if python --version 2>&1 | grep -q "Python 3"; then
            echo "python"
        else
            print_error "Python 3 is required but not found"
            return 1
        fi
    else
        print_error "Python 3 is required but not found"
        print_error "Please install Python 3.7+ and try again"
        return 1
    fi
}

# Main setup function
main() {
    echo "============================================================"
    echo "🧠 MOTOROLA DREAM MACHINE - AUTOMATED SETUP"
    echo "============================================================"
    echo "Setting up EEG-to-Robot Control System..."
    echo "============================================================"
    
    # Find Python
    print_status "Checking Python installation..."
    PYTHON_CMD=$(find_python)
    if [ $? -ne 0 ]; then
        exit 1
    fi
    print_success "Found Python: $PYTHON_CMD"
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    print_status "Python version: $PYTHON_VERSION"
    
    # Create virtual environment
    print_status "Creating virtual environment..."
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            $PYTHON_CMD -m venv venv
            print_success "Virtual environment recreated"
        else
            print_status "Using existing virtual environment"
        fi
    else
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    python -m pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        print_success "Dependencies installed from requirements.txt"
    else
        print_status "Installing core dependencies manually..."
        pip install numpy scipy pyyaml pandas matplotlib scikit-learn
        print_status "Installing optional dependencies..."
        pip install torch mne joblib seaborn plotly || print_warning "Some optional packages failed to install"
        print_success "Core dependencies installed"
    fi
    
    # Create directories
    print_status "Creating necessary directories..."
    mkdir -p config logs data models output scripts
    print_success "Directory structure created"
    
    # Run Python setup
    print_status "Running Python setup script..."
    python scripts/setup.py
    
    # Create activation script
    print_status "Creating activation script..."
    cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activation script for Motorola Dream Machine

if [ -d "venv" ]; then
    echo "🧠 Activating Motorola Dream Machine environment..."
    source venv/bin/activate
    echo "✅ Environment activated!"
    echo ""
    echo "Quick commands:"
    echo "  python3 scripts/quick_start.py --demo    # Run demo"
    echo "  python3 scripts/quick_start.py --test    # Run tests"
    echo "  python3 scripts/quick_start.py           # Full system"
    echo ""
else
    echo "❌ Virtual environment not found. Run ./setup.sh first."
fi
EOF
    chmod +x activate_env.sh
    print_success "Created activate_env.sh for easy environment activation"
    
    # Final instructions
    echo ""
    echo "============================================================"
    print_success "🎉 SETUP COMPLETE!"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Activate the environment:"
    echo "   source venv/bin/activate"
    echo "   # or use: ./activate_env.sh"
    echo ""
    echo "2. Run a quick demo:"
    echo "   python3 scripts/quick_start.py --demo"
    echo ""
    echo "3. Update config/emotiv.yaml with your Emotiv credentials"
    echo ""
    echo "4. Start the full system:"
    echo "   python3 scripts/quick_start.py"
    echo ""
    echo "============================================================"
    
    # Test installation
    print_status "Testing installation..."
    if python -c "import numpy, scipy, yaml, pandas; print('✅ Core dependencies OK')"; then
        print_success "Installation test passed!"
    else
        print_warning "Some dependencies may have issues"
    fi
    
    echo ""
    print_success "Setup completed successfully! 🚀"
}

# Run main function
main "$@"
