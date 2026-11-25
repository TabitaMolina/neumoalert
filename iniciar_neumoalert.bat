@echo off
title NEUMOALERT - Plataforma IA Multimodal
cd /d "%~dp0"
echo Iniciando NEUMOALERT...
".venv\Scripts\streamlit.exe" run app.py
pause
