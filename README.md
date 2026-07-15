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

## Change the RAG dataset

The web UI has a dataset picker in the sidebar. Select a `.csv`, `.xls`, or `.xlsx` file, then click **Use for RAG**. The API saves the upload in `uploaded_datasets/`, rebuilds `data/manga.db`, `data/manga.faiss`, and `data/manga_mapping.pkl`, then swaps the live RAG controller to the rebuilt dataset.

You can also call the endpoint directly:

```powershell
curl.exe -X POST http://127.0.0.1:8000/dataset -F "dataset=@dataset/manga_dataset.csv"
```

Dataset files must include the manga columns used by the current RAG pipeline, including `mal_id`, `title`, `title_english`, `score`, `authors`, `genres`, `themes`, and `synopsis`.
