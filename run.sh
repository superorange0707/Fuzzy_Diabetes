#!/bin/bash

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON=python3
elif command -v python &>/dev/null; then
    PYTHON=python
else
    echo "Error: Python is not installed. Please install Python 3.x."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ] || [ ! -f "venv/bin/activate" ]; then
    echo "Creating/recreating virtual environment..."
    rm -rf venv
    $PYTHON -m venv venv
    
    # Check if venv was created successfully
    if [ ! -f "venv/bin/activate" ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Verify activation was successful
if [ ! "$VIRTUAL_ENV" ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi

# Check if pip is working
if ! command -v pip &>/dev/null; then
    echo "Error: pip is not available in the virtual environment."
    exit 1
fi

# Upgrade pip and install fundamental packages first
echo "Upgrading pip and installing fundamental packages..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create temp model files for demonstration
echo "Creating sample model files if needed..."
if [ ! -d "Model" ]; then
    mkdir -p Model/Comparison
fi

# Run the application
echo "Starting Fuzzy Diabetes application..."
cd app
python -m streamlit run app.py

# Deactivate virtual environment when done
deactivate 