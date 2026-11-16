@echo off
rem ==========================================================================
rem  Panel "Rozpoznawanie twarzy" - uruchamianie dwuklikiem.
rem
rem  Ten plik robi dokladnie to samo, co w terminalu:
rem      .venv\Scripts\python.exe gui\main.py
rem  czyli otwiera natywne okno panelu (pywebview + Bootstrap).
rem
rem  DLA UCZNIOW: nie trzeba znakowac konsoli ani aktywowac .venv -
rem  plik sam szuka wlasciwego Pythona, a okno zamyka sie razem z panelem.
rem ==========================================================================

setlocal EnableExtensions
chcp 65001 >nul 2>&1
title Rozpoznawanie twarzy - panel
cd /d "%~dp0"

rem --- 1. Najpierw Python z wirtualnego srodowiska projektu (.venv) --------
set "PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%PYTHON%" goto uruchom

rem --- 2. Brak .venv: launcher "py" (instalator z python.org) -------------
set "PYTHON=py -3"
where py >nul 2>&1
if not errorlevel 1 goto uruchom

rem --- 3. Na koncu zwykly "python" z PATH ----------------------------------
set "PYTHON=python"
where python >nul 2>&1
if not errorlevel 1 goto uruchom
goto brak_pythona

:uruchom
echo Uruchamianie panelu...
echo (to okno mozna zminimalizowac; zamknie sie razem z panelem)
echo.

rem Sciezka do .venv moze zawierac spacje, dlatego jest w cudzyslowie.
if /i "%PYTHON%"=="py -3" goto uruchom_py
"%PYTHON%" "gui\main.py"
goto po_uruchomieniu

:uruchom_py
py -3 "gui\main.py"

:po_uruchomieniu
set "KOD=%errorlevel%"
if "%KOD%"=="0" exit /b 0

echo.
echo ==========================================================================
echo  Panel zakonczyl sie bledem (kod %KOD%).
echo.
echo  Najczestsze przyczyny:
echo    - brak bibliotek:  .venv\Scripts\pip install -r requirements.txt
echo    - kamera uzywana przez inny program (zamknij inne aplikacje),
echo    - brak WebView2 (awaryjnie: .venv\Scripts\python.exe gui\tkinter_app.py).
echo ==========================================================================
pause
exit /b %KOD%

:brak_pythona
echo.
echo ==========================================================================
echo  Nie znaleziono Pythona.
echo.
echo  1. Zainstaluj Python 3.10+ z python.org (zaznacz "Add Python to PATH").
echo  2. W folderze projektu wykonaj:
echo         python -m venv .venv
echo         .venv\Scripts\pip install -r requirements.txt
echo  3. Kliknij ten plik jeszcze raz.
echo ==========================================================================
pause
exit /b 1
