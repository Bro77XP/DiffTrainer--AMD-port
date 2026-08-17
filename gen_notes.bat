@echo off
setlocal

echo ======================================================================
echo  WARNING: This script auto-generates note_seq/note_dur from the audio
echo  f0 contour (one note per phone). It is a pitch-following approximation,
echo  NOT real musical score annotation. Variance training will run, but you
echo  will NOT get true note-level pitch control at inference. For that you
echo  still need real MIDI/score notes in transcriptions.csv.
echo ======================================================================
echo.

cd /d "%~dp0"
set PYTHONPATH=%CD%
call ..\venv\Scripts\activate.bat
python scripts/gen_notes.py --config %CD%\configs\variance.yaml
pause
