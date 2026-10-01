from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import requests

from rag import search_documents


app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "static" / "index.html"
    )


@app.post("/chat")
def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a message."
        )

    try:
        documents = search_documents(
            request.message,
            n_results=4
        )

        context = "\n\n".join(documents)

        prompt = f"""
You are an AI assistant that answers questions using the provided document.

Use the document context below to answer the user's question.

If the answer is not available in the document context,
say that the information is not available in the provided document.

Document Context:
{context}

User Question:
{request.message}
"""

        response = requests.post(
            "http://127.0.0.1:11434/api/chat",
            json={
                "model": "llama3.2:3b",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        return {
            "reply": data["message"]["content"]
        }

    except requests.RequestException as e:

        print("OLLAMA ERROR:", str(e))

        raise HTTPException(
            status_code=502,
            detail="Cannot connect to Ollama. Make sure Ollama is running."
        )

    except Exception as e:

        print("RAG ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )