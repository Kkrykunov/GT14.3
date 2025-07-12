#!/bin/bash

# GT14 WhaleTracker v14.3 - Launch Script for Linux/Mac
# This script sets up the environment and runs the WhaleTracker application

# Set UTF-8 locale
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
export PYTHONUTF8=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GT14 WhaleTracker v14.3${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${YELLOW}Python version: $PYTHON_VERSION${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update requirements
echo -e "${YELLOW}Installing/updating requirements...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Run the main application
echo -e "${GREEN}Starting GT14 WhaleTracker...${NC}"
echo -e "${GREEN}========================================${NC}"
python GT14_v14_3_FINAL.py "$@"

# Deactivate virtual environment
deactivate