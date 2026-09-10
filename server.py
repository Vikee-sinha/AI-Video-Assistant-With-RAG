"""
Afterword — backend server

Wraps the existing pipeline (utils.audio_processor, core.Transcribe,
core.summerizer, core.vector_db, core.rag_engine) behind a small FastAPI
app so the frontend (static/index.html) can drive it over HTTP.

Run with:
    pip install fastapi uvicorn python-multipart
    uvicorn server:app --reload --port 8000

Then open http://localhost:8000/
"""

import os
import shutil
import tempfile
import uuid
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from utils.audio_processor import process_input
from core.Transcribe import transcribe_all
from core.summerizer import analyze_meeting, parse_analysis
from core.vector_db import get_vector_store
from core.rag_engine import load_rag_chain, ask_question

load_dotenv()

app = FastAPI(title="Afterword")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> { "rag_chain": ..., "transcript": ... }
# This is intentionally simple (no DB). Fine for a single-user / local tool;
# swap for redis or similar if this needs to run multi-user in production.
SESSIONS: dict[str, dict] = {}

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "afterword_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class AskRequest(BaseModel):
    session_id: str
    question: str


def _save_upload(file: UploadFile) -> str:
    """Persist an uploaded file to disk and return its path."""
    suffix = os.path.splitext(file.filename or "")[1]
    dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest_path


@app.post("/api/process")
async def process(
    language: str = Form("english"),
    youtube_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    if not youtube_url and not file:
        raise HTTPException(
            status_code=400,
            detail="Provide either a youtube_url or a file.",
        )

    saved_path = None
    try:
        source = youtube_url.strip() if youtube_url else _save_upload(file)
        if file:
            saved_path = source

        # 1. Normalize input (download / convert / chunk as needed)
        audio_chunks = process_input(source)

        # 2. Transcribe all audio chunks
        transcript = transcribe_all(audio_chunks, language)

        # 3. Single analysis call -> title / summary / action items / etc.
        analysis = analyze_meeting(transcript)
        meeting_data = parse_analysis(analysis)

        # 4. Build the vector store for this transcript
        get_vector_store(transcript)

        # 5. Load a RAG chain bound to that store
        rag_chain = load_rag_chain()

        # 6. Answer a default framing question up front
        default_question = "What are the key takeaways from the meeting?"
        initial_answer = ask_question(rag_chain, default_question)

        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = {
            "rag_chain": rag_chain,
            "transcript": transcript,
        }

        return {
            "session_id": session_id,
            "title": meeting_data.get("title"),
            "summary": meeting_data.get("summary"),
            "action_items": meeting_data.get("action_items"),
            "key_decisions": meeting_data.get("key_decisions"),
            "questions": meeting_data.get("questions"),
            "transcript": transcript,
            "initial_question": default_question,
            "initial_answer": initial_answer,
        }
    except Exception as exc:  # surface pipeline failures to the UI
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Keep the upload only if you want to reprocess later; otherwise clean up.
        if saved_path and os.path.exists(saved_path):
            pass  # left on disk deliberately in case transcription needs re-runs


@app.post("/api/ask")
async def ask(payload: AskRequest):
    session = SESSIONS.get(payload.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Process a meeting first.",
        )
    try:
        answer = ask_question(session["rag_chain"], payload.question)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---- static frontend -------------------------------------------------

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    # Render.com sets the PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
    print(f"Server running on port {port}")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))
