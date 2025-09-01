@echo off
REM venv
call ""

REM Videos Folder
set VIDEO_FOLDER=

REM Output Folder
set OUTPUT_FOLDER=

REM Trained Model
set MODEL_PATH=

REM Process all videos .mp4
for %%f in ("%VIDEO_FOLDER%\*.mp4") do (
    echo Processing %%f ...
    py "" "%MODEL_PATH%" "%%f" "%OUTPUT_FOLDER%"
)

echo All videos processed ;)
pause
