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
APEX_ROOT="$PWD"

# Download llama.cpp b11157 binaries
echo "Downloading llama.cpp b11157 binaries..."
mkdir -p ~/llama
cd ~/llama
wget https://github.com/ggml-org/llama.cpp/releases/download/b11157/llama-b11157-bin-ubuntu-x64.tar.gz

# Extract the tarball
echo "Extracting llama.cpp binaries..."
tar -xzf llama-b11157-bin-ubuntu-x64.tar.gz

# Locate llama-quantize binary within the extracted tree
LLAMA_QUANTIZE=$(find ~/llama -type f -name 'llama-quantize' | head -n 1)
if [ -z "$LLAMA_QUANTIZE" ]; then
    echo "ERROR: llama-quantize binary not found in extracted tarball"
    exit 1
fi

# LLAMA_CPP_DIR is the directory that directly contains llama-quantize
LLAMA_CPP_DIR=$(dirname "$LLAMA_QUANTIZE")

# Write .env for the apex-quant project
touch "$APEX_ROOT/.env"
printf 'LLAMA_QUANTIZE=%s\n' "$LLAMA_QUANTIZE" >> "$APEX_ROOT/.env"
printf 'LLAMA_CPP_DIR=%s\n' "$LLAMA_CPP_DIR" >> "$APEX_ROOT/.env"

cd $APEX_ROOT

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

echo "✅ Setup completed successfully!"
echo "You can now run the application using: python3 main.py"
