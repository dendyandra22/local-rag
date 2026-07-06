import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rag_module.rag_controller import RAGController

app = FastAPI()
ragc = RAGController(db_name='manga', timestamp='2026-06-24', inference_type=None, retrain_model=False)

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_UI_DIR = BASE_DIR / "web_ui"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
MAX_HISTORY_MESSAGES = 40

app.mount("/static", StaticFiles(directory=WEB_UI_DIR), name="static")


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]




def _safe_session_id(session_id: str | None) -> str:
    if not session_id:
        return uuid4().hex

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    return session_id


def _history_path(session_id: str) -> Path:
    return CHAT_HISTORY_DIR / f"{session_id}.json"


def _load_history(session_id: str) -> list[dict]:
    path = _history_path(session_id)
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    messages = data.get("messages", [])
    if not isinstance(messages, list):
        return []

    return [
        {
            "role": item.get("role"),
            "content": item.get("content", ""),
        }
        for item in messages
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]


def _save_history(session_id: str, messages: list[dict]) -> None:
    CHAT_HISTORY_DIR.mkdir(exist_ok=True)
    payload = {
        "session_id": session_id,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages[-MAX_HISTORY_MESSAGES:],
    }

    path = _history_path(session_id)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def chat_ui():
    return FileResponse(WEB_UI_DIR / "index.html")


@app.get("/history/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str):
    session_id = _safe_session_id(session_id)
    return {
        "session_id": session_id,
        "messages": _load_history(session_id),
    }


@app.post("/chat")
async def chat_api(request: ChatRequest):
    session_id = _safe_session_id(request.session_id)
    history = _load_history(session_id)

    def stream_and_store():
        chunks = []
        for chunk in ragc.response_handler(request.message, verbose=1, stream=True, chat_history=history):
            chunks.append(chunk)
            yield chunk

        assistant_message = ''.join(chunks)
        updated_history = history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": assistant_message},
        ]
        _save_history(session_id, updated_history)

    return StreamingResponse(
        stream_and_store(),
        media_type="text/plain",
        headers={"X-Session-Id": session_id},
    )


# run on cmd => fastapi dev rag_module/rag_api.py

