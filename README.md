# Local RAG
A lightweight, local RAG prototype for structured tabular data (.csv and .xlsx) powered by FastAPI and Qwen2.5-3B-Instruct.

## Features
Query structured datasets using natural language—including data lookups, aggregations (AVG, SUM, COUNT), filtering, and full-text search.

### Example queries:

- "Find average user rating grouped by primary genre where developer is Epic Wallet Ltd"

- "Show me paid games with a rating above 3.5"

- "What is ReLife manga?"

- "Show me 5 titles containing the word 'OJK'"

- "How many unique values are in Media Name?"

## Getting Started
### 1. Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```
### 2. Add Dataset 

Place your `.csv` or `.xlsx` file inside the `dataset/` directory. 

### 3. Create Config File 

Create a configuration file at `dataset/<your_rag_name>.ini` using this structure:

```text
[RAG]
rag_name = <your_rag_name>

[DATASET]
path = <filename.csv_or_filename.xlsx>

; Optional: Limit rows loaded into SQL (default: loads all rows)
; max_rows = 3000

; Column to use as index/primary key
index_column = <column_name>

; Main searchable columns (null values will be cleaned; comma-separated)
main_columns = <col1>, <col2>, <col3>

; Optional: Columns to drop (comma-separated)
drop_columns = <col_a>, <col_b>
```

### 4. Build & Start the RAG Server

First-time build (creates SQLite database), automatically start up after completed build RAG:

```python
python run.py -c <your_rag_name>
```
If you already create RAG before, then you can Start API server (existing setup):
```python
python run.py -n <your_rag_name>
```

### 5. Open Web UI
Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## Limitation
- 3B Model Constraints: Highly complex or nested multi-step questions may occasionally fail or hallucinate.

- Inference Speed: Running strictly on CPU will result in slower response times compared to GPU execution.

- Manual Configuration: Requires explicit schema mapping in an .ini file before running.

- Basic Interface: Minimal UI with limited controls for switching datasets on the fly.
