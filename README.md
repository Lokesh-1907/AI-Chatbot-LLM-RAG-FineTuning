# AI Chatbot with Ollama

Uses Python, FastAPI, Ollama and Llama 3.2 3B.

## Setup

1. Install Ollama for Windows: https://ollama.com/download/windows
2. Open PowerShell:
   ollama pull llama3.2:3b
3. Test:
   ollama run llama3.2:3b
4. Open this project in VS Code.
5. In the folder containing main.py:
   python -m pip install -r requirements.txt
6. Run:
   python -m uvicorn main:app --reload
7. Open:
   http://127.0.0.1:8000

Do not use Live Server for this version.

Ollama and the Llama model are not included in this ZIP because they are separate software/downloads.
