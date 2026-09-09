import argparse
import uvicorn
import os

# CLI Parser
parser = argparse.ArgumentParser(description="RAG API server")
parser.add_argument("--rag_name", required=True, help="RAG name.")
parser.add_argument("--rebuild", type=str, default="False", help="(Optional) Set to True to rebuild the RAG. Default is False.")
parser.add_argument("--max_history_limit", type=int, default=40, help="(Optional) Maximum number of history messages saved.")

args = parser.parse_args()

os.environ["RAG_NAME"] = args.rag_name
os.environ["RAG_REBUILD"] = args.rebuild.lower()
os.environ["MAX_HISTORY_MESSAGES"] = str(args.max_history_limit)

if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run(
          "rag_module.rag_api:app",
          host="127.0.0.1",
          port=8000,
          reload=True,  # Matches the behavior of `fastapi dev`
      )