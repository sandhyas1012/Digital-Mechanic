@echo off
echo ===================================================
echo    Digital Mechanic - Full System Initialization
echo ===================================================

echo.
echo [1/3] Ingesting Ford Workshop Manuals (PDFs)...
call .venv\Scripts\python.exe ingest_pdfs.py

echo.
echo [2/3] Fetching and Processing YouTube Videos...
call .venv\Scripts\python.exe youtube_scraper.py

echo.
echo [3/3] Starting the Digital Mechanic Web Interface...
call .venv\Scripts\python.exe app.py

pause
