@echo off
setlocal EnableDelayedExpansion

REM ===== CONFIG =====
set prefix=Healthy

REM ===== STEP 1: Rename everything to unique internal names =====
set i=1
for %%f in (*.jpg *.png *.jpeg) do (
    ren "%%f" "__tmp_!i!%%~xf"
    set /a i+=1
)

REM ===== STEP 2: Final clean rename (starts from 1) =====
set i=1
for %%f in (__tmp_*.jpg __tmp_*.png __tmp_*.jpeg) do (
    ren "%%f" "%prefix%_!i!%%~xf"
    set /a i+=1
)

echo =====================================
echo Done! Images renamed safely.
echo Prefix: %prefix%
echo Start index: 1
echo =====================================
pause
