@echo off
REM Setup script for Smart Car Project (Windows)

python -m venv venv
call venv\Scripts\activate
pip install --upgrade pip
pip install gradio pyserial
echo Setup complete. You can now run run_project.bat
pause
