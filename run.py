import argparse
import uvicorn
import os
from pathlib import Path

from rag_module.rag_api import create_api
from rag_module.rag_controller_llama import RAGAction
from rag_module.config import generate_config


# CLI Parser
def main(args: argparse.Namespace):
    if args.create:
        print(f"🔨 Generating config file: {args.create}.ini -> {args.create}_config.ini")
        generate_config(args.create)
        print(f"🔨 Creating RAG setup for: {args.create}...")
        RAGAction(rag_name=args.create).create_rag_from_dataset()

    if args.rag_name or args.create:
        print(f"Start up RAG server...")
        rag_name = args.rag_name if args.rag_name else args.create
        app = create_api(rag_name=rag_name)
        uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RAGLocal',
        description='Start up/create local RAG server')
    parser.add_argument('-n', '--rag_name', type=str, help="RAG name will be started")
    parser.add_argument('-c', '--create', type=str, help="Create new RAG local")
    # parser.add_argument('--max_history', type=int, help="Maximum number of history messages will be inputted to LLM")
    # parser.add_argument('-v', '--verbose',
    #                     action='store_true')  # on/off flag
    args = parser.parse_args()
    print("run args param:",args.rag_name, args.create)

    main(args)
