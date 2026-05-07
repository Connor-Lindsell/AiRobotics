@echo off
echo Starting TensorBoard...
echo Open http://localhost:6006 in your browser.
echo Press Ctrl+C to stop.
echo.
call "%~dp0venv\Scripts\activate.bat"
tensorboard --logdir "%~dp0logs"
