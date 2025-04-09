@echo off
REM Setup script for Smart Car Project (Windows)

python -m venv smartcar-env
call smartcar-env\Scripts\activate
pip install --upgrade pip
pip install gradio pyserial
echo Setup complete. You can now run run_project.bat
pause
