#!/bin/bash
set -euo pipefail

# Function to check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Update package list
echo "Updating package list..."
sudo apt-get update

# Install tmux, wget, and unzip
echo "Installing tmux, wget, and unzip..."
sudo apt-get install -y tmux wget unzip python3 python3-pip python3-venv

cd ~/

# Download apex-quant
echo "Downloading apex-quant from GitHub..."
wget https://github.com/DrMoriarty/apex-quant/archive/refs/heads/main.zip

# Extract the archive
echo "Extracting archive..."
unzip main.zip
cd apex-quant-main

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

echo "✅ Setup completed successfully!"
echo "You can now run the application using: python3 main.py"
