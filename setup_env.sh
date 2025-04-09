#!/bin/bash
# Setup script for Smart Car Project

set -e
python3 -m venv smartcar-env
source smartcar-env/bin/activate
pip install --upgrade pip
pip install gradio pyserial
echo "Setup complete. You can now run run_project.sh"
