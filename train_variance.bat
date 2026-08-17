@echo off
setlocal

echo ======================================================================
echo  Variance training.
echo  NOTE: train into a SEPARATE work dir from the acoustic model. The
echo  variance model has a different architecture and CANNOT resume an
echo  acoustic checkpoint (model_ckpt_steps_*.ckpt). --reset only resets
echo  hparams, NOT checkpoints, so a fresh exp_name is required.
echo ======================================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%CD%
call ..\venv\Scripts\activate.bat
python scripts/train.py --config %CD%\configs\variance.yaml --exp_name %CD%\checkpoints\variance
pause
