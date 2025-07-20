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
