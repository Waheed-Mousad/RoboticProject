#!/bin/bash

# === 1. Wait for internet ===
echo "📡 Checking internet connection..."
until ping -c1 google.com &>/dev/null; do
    echo "❌ No internet. Retrying in 2 seconds..."
    sleep 2
done
echo "✅ Internet connection detected."

# === 2. Pull latest GitHub repo ===
echo "📥 Pulling latest changes from GitHub..."
cd /home/w/Documents/GitHub/RoboticProject || exit 1
git reset --hard HEAD       # Force overwrite local changes
git pull origin main        # Replace 'main' with your branch if needed

# === 3. Launch the Gradio app ===
echo "🚀 Launching Gradio app..."
source venv/bin/activate  # If using virtualenv
python3 GradioApp.py
