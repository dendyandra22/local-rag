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
# from rag_module.rag_controller import RAGAction
from rag_module.rag_controller_llama import RAGAction
from database_module.tabular_data import SUPPORTED_DATASET_EXTENSIONS

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_UI_DIR = BASE_DIR / "web_ui"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
DATA_DIR = BASE_DIR / "data"
MAX_HISTORY_MESSAGES = 40

# RAG source setting below:
RAG_NAME = "apple" # manga apple
RAG_TIMESTAMP = "2026-06-24"
RAG_INFERENCE_TYPE = None
RAG_REBUILD = False

ragc = RAGAction(
    rag_name=RAG_NAME,
    # nlu_model_name=RAG_TIMESTAMP,
    # inference_type=RAG_INFERENCE_TYPE,
)
rag_lock = threading.Lock()
active_dataset = {
    "filename": None,
    "uploaded_at": None,
    "db_name": None,
}

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


class DatasetStatusResponse(BaseModel):
    filename: str | None
    uploaded_at: str | None
    db_name: str | None
    supported_extensions: list[str]


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


# def _safe_upload_filename(file_path: str) -> Path:
#     if file_path == "":
#         raise HTTPException(status_code=400, detail="File path is empty")
#
#     original = Path(file_path)
#     extension = original.suffix.lower()
#
#     if extension not in SUPPORTED_DATASET_EXTENSIONS:
#         supported = ", ".join(sorted(SUPPORTED_DATASET_EXTENSIONS))
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported dataset file type. Use one of: {supported}",
#         )
#
#     # stem = Path(original).stem
#     # stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "dataset"
#     # stem = stem.encode('utf-8').hex()
#     # return f"{stem}"
#     return original


# def _rebuild_rag_from_dataset(source_path: Path, save_path: Path) -> None:
#     global ragc
#
#     ragc = RAGController()




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


@app.get("/dataset", response_model=DatasetStatusResponse)
def get_dataset_status():
    return {
        **active_dataset,
        "supported_extensions": sorted(SUPPORTED_DATASET_EXTENSIONS),
    }


@app.post("/dataset", response_model=DatasetUpload)
async def upload_dataset(item: DatasetUpload):

    # source_path = _safe_upload_filename(item.file_path or "")
    # DATA_DIR.mkdir(exist_ok=True)
    # # saved_path = DATA_DIR / source_path.name
    #
    # print('source_path', source_path)
    #
    #
    # try:
    #         with rag_lock:
    #             ragc = _rebuild_rag_from_dataset(source_path, saved_path)
    #     #
    #     active_dataset.update({
    #         "filename": str(source_path.name),
    #         "uploaded_at": datetime.now(timezone.utc).isoformat(),
    #         "db_name": str(source_path.stem),
    #     })

    # except ValueError as exc:
    #     # saved_path.unlink(missing_ok=True)
    #     raise HTTPException(status_code=400, detail=str(exc)) from exc
    # except Exception as exc:
    #     # saved_path.unlink(missing_ok=True)
    #     raise HTTPException(status_code=500, detail=f"Could not load dataset: {exc}") from exc
    #
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

        # for chunk in active_ragc.response_handler(request.message, verbose=True, stream=True, chat_history=None):
        #     chunks.append(chunk)
        #     yield chunk
        for chunk in active_ragc.response_handler_with_tool(request.message, verbose=True, stream=True, chat_history=history, chat_history_limit=4):
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
