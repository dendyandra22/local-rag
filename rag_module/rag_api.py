import json
import re
import threading
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
from rag_module.rag_controller_llama import RAGAction
from rag_module.config import load_config




def create_api(rag_name: str, verbose: bool = False):
    config = load_config(f"{rag_name}_config.ini")

    base_dir = Path(__file__).resolve().parent.parent
    web_ui_dir = base_dir / "web_ui"
    
    rag_name = config["RAG"].get('rag_name')
    max_history_messages = int(config["LLM"].get('max_history_messages'))
    chat_history_dir = base_dir / "chat_history" / rag_name

    app = FastAPI()
    
    ragc = RAGAction(
        rag_name=rag_name,
        verbose=verbose
    )
    rag_lock = threading.Lock()
    active_dataset = {
        "filename": None,
        "uploaded_at": None,
        "db_name": None,
    }
    
    app.mount("/static", StaticFiles(directory=web_ui_dir), name="static")
    
    
    class ChatMessage(BaseModel):
        role: Literal["user", "assistant"]
        content: str
    
    class ChatRequest(BaseModel):
        message: str
        session_id: str | None = None
    
    
    class ChatHistoryResponse(BaseModel):
        session_id: str
        messages: list[ChatMessage]
    
    class DatasetUpload(BaseModel):
        file_path: str
        uploaded_at: str | None = None
    
    
    
    def _safe_session_id(session_id: str | None) -> str:
        if not session_id:
            return uuid4().hex
    
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", session_id):
            raise HTTPException(status_code=400, detail="Invalid session_id")
    
        return session_id
    
    
    def _history_path(session_id: str) -> Path:
        return chat_history_dir / f"{session_id}.json"
    
    
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
        chat_history_dir.mkdir(exist_ok=True)
        payload = {
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": messages[-max_history_messages:],
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
        return FileResponse(web_ui_dir / "index.html")
    
    
    @app.get("/history/{session_id}", response_model=ChatHistoryResponse)
    def get_chat_history(session_id: str):
        session_id = _safe_session_id(session_id)
        return {
            "session_id": session_id,
            "messages": _load_history(session_id),
        }
    
    @app.post("/dataset", response_model=DatasetUpload)
    async def upload_dataset(item: DatasetUpload):
        return {
            "file_path": "Unavailable",
            "uploaded_at": active_dataset["uploaded_at"],
        }
    
    
    @app.post("/chat")
    async def chat_api(request: ChatRequest):
        session_id = _safe_session_id(request.session_id)
        history = _load_history(session_id)
    
        def stream_and_store():
            chunks = []
            with rag_lock:
                active_ragc = ragc

            for chunk in active_ragc.response_handler_with_tool(request.message, chat_history=history, chat_history_limit=2):
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
    
    return app
