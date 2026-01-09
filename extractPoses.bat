@echo off
REM This script extracts a series of 2D poses for every .mp4 file in the specified directory,
REM but only if the corresponding result folder does not already exist.

setlocal enabledelayedexpansion
set "CLIPDIR=Data\Shots"
set "RESULTDIR=Data\PosesRaw"
set KMP_DUPLICATE_LIB_OK=TRUE

for %%F in ("%CLIPDIR%\*.mp4") do (
    set "FILENAME=%%~nF"
    set "RESULTFOLDER=%RESULTDIR%\%%~nF_Sports2D"
    if exist "!RESULTFOLDER!\" (
        echo Skipping %%F, result folder already exists.
    ) else (
        echo Processing %%F
        sports2d --nb_persons_to_detect 1 --person_ordering_method greatest_displacement --first_person_height 1.80 --visible_side none --show_graphs false --result_dir Data\PosesRaw --show_realtime_results false --video_input "%%F"
    )
)

endlocal