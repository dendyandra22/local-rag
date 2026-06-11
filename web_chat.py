import cgi
import json
import mimetypes
import re
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from rag_chat_adapter import generate_response


ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
UPLOAD_DIR = ROOT / "uploads"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".log", ".py", ".html", ".css", ".js"}


def _safe_name(filename):
    name = Path(filename or "upload").name
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name).strip()
    return name or "upload"


def _unique_path(directory, filename):
    base = Path(_safe_name(filename))
    stem = base.stem or "upload"
    suffix = base.suffix
    candidate = directory / f"{stem}{suffix}"
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _read_text_preview(path, limit=6000):
    suffix = path.suffix.lower()
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="replace")[:limit]

    if suffix == ".pdf":
        for module_name in ("pypdf", "PyPDF2"):
            try:
                module = __import__(module_name)
                reader_cls = getattr(module, "PdfReader")
                reader = reader_cls(str(path))
                pages = []
                for page in reader.pages[:6]:
                    pages.append(page.extract_text() or "")
                return "\n".join(pages)[:limit]
            except Exception:
                continue
        return ""

    return ""


class ChatHandler(BaseHTTPRequestHandler):
    server_version = "LocalRAGChat/0.1"

    def do_GET(self):
        if self.path == "/":
            self._serve_file(WEB_DIR / "index.html")
            return

        if self.path == "/api/files":
            self._json_response({"files": self._uploaded_files()})
            return

        requested = unquote(self.path.lstrip("/"))
        path = (WEB_DIR / requested).resolve()
        if WEB_DIR.resolve() not in path.parents and path != WEB_DIR.resolve():
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        self._serve_file(path)

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
            return

        if self.path == "/api/upload":
            self._handle_upload()
            return

        self.send_error(404)

    def log_message(self, format, *args):
        print("%s - - %s" % (self.address_string(), format % args))

    def _serve_file(self, path):
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _uploaded_files(self):
        UPLOAD_DIR.mkdir(exist_ok=True)
        files = []
        for path in sorted(UPLOAD_DIR.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            if not path.is_file():
                continue
            files.append({
                "name": path.name,
                "size": path.stat().st_size,
                "path": str(path),
                "text_preview": _read_text_preview(path, limit=2000),
            })
        return files

    def _handle_chat(self):
        try:
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            history = payload.get("history", [])
            selected_files = payload.get("files", [])
            if not message:
                self._json_response({"error": "Message is required"}, status=400)
                return

            known_files = {item["name"]: item for item in self._uploaded_files()}
            files = [known_files[name] for name in selected_files if name in known_files]
            response = generate_response(message, history=history, files=files)
            self._json_response(response)
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=500)

    def _handle_upload(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > MAX_UPLOAD_BYTES:
                self._json_response({"error": "File is larger than 25 MB"}, status=413)
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            field = form["file"] if "file" in form else None
            if field is None or not field.filename:
                self._json_response({"error": "No file uploaded"}, status=400)
                return

            UPLOAD_DIR.mkdir(exist_ok=True)
            destination = _unique_path(UPLOAD_DIR, field.filename)
            with destination.open("wb") as handle:
                shutil.copyfileobj(field.file, handle)

            payload = {
                "name": destination.name,
                "size": destination.stat().st_size,
                "path": str(destination),
                "text_preview": _read_text_preview(destination, limit=2000),
            }
            self._json_response({"file": payload})
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=500)


def main(host="127.0.0.1", port=8000):
    WEB_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((host, port), ChatHandler)
    print(f"Local RAG chat UI running at http://{host}:{port}")
    print("Set RAG_CHAT_HANDLER=module:function to connect your Python RAG app.")
    server.serve_forever()


if __name__ == "__main__":
    main()
