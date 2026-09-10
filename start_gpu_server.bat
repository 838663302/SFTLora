@echo off
title Local OpenVINO GPU/NPU Server
echo ========================================================
echo   Starting Local OpenVINO API Server on Intel Arc GPU
echo   API Base URL: http://127.0.0.1:8000/v1
echo ========================================================
C:\Users\azt1szh\.conda\envs\SFTLora\python.exe c:\Users\azt1szh\Desktop\set\SFTLora\api_server.py --device GPU --port 8000
pause
