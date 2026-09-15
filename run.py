import argparse
import uvicorn
import os
from pathlib import Path

from rag_module.rag_api import create_api
from rag_module.rag_controller_llama import RAGAction
from rag_module.config import generate_config
from util.logger import print_log


# CLI Parser
def main(arguments: argparse.Namespace):
    if arguments.create:
        print_log(f"🔨 Generating config file: {arguments.create}.ini -> {arguments.create}_config.ini")
        generate_config(arguments.create)
        print_log(f"🔨 Creating RAG setup for: {arguments.create}...")
        RAGAction(rag_name=arguments.create, verbose=arguments.verbose).create_rag_from_dataset()

    if arguments.rag_name or arguments.create:
        print_log(f"Start up RAG server...")
        rag_name = arguments.rag_name if arguments.rag_name else arguments.create
        app = create_api(rag_name=rag_name, verbose=arguments.verbose)
        uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='RAGLocal',
        description='Start up/create local RAG server')
    parser.add_argument('-n', '--rag_name', type=str, help="RAG name will be started")
    parser.add_argument('-c', '--create', type=str, help="Create new RAG local")
    parser.add_argument('-v', '--verbose',
                        action='store_true')  # on/off flag
    args = parser.parse_args()

    main(args)
