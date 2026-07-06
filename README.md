# manga-hybrid-rag
RAG chatbot about manga

## Run the local web UI

1. Start Ollama and make sure the `gemma3:4b` model is available.
2. Install the Python dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Start the FastAPI app:

   ```powershell
   fastapi dev rag_module/rag_api.py
   ```

4. Open the chat UI:

   ```text
   http://127.0.0.1:8000
   ```

Chat sessions are stored as JSON files in `chat_history/`, and recent messages are passed back into the model so follow-up questions can refer to earlier turns.
