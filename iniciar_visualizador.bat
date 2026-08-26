@echo off
echo Starting ASL translator with filter visualizer...
call .\venv312\Scripts\activate.bat
python realtime_translator.py --activations
if errorlevel 1 pause
