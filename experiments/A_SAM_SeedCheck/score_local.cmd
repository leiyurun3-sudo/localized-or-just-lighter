@echo off
setlocal
if "%~1"=="" (
  echo Drag one returned A_SAM_seed ZIP onto this file, or pass its path.
  exit /b 2
)
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
set "USE_TF=0"
set "USE_FLAX=0"
set "PYTHONPATH=D:\DevCache\doselens_metrics\deps;D:\DevCache\tracememory_metrics\packages;%PYTHONPATH%"
python -B "%~dp0study\score.py" "%~f1"
exit /b %errorlevel%

