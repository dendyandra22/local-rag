import argparse
import configparser
from pathlib import Path
import re

from util.logger import print_log

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR / "dataset" # temp project dir is dataset

def config_sep2list(value: str):
    if value in ["",None,"None"]:
        return []

    result = [
        f.strip()
        for f in value.split(", ")
    ]

    return result

def load_config(config_name: Path | str):
    if isinstance(config_name, str):
        config_name = Path(config_name)
    if config_name.suffix == "":
        config_name = config_name.with_suffix(".ini")

    config_path = Path(PROJECT_DIR / config_name)
    if not config_path.is_file():
        print(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"{config_path.stem}{config_path.suffix} not found in dataset directory. Please create RAG first.")

    config = configparser.ConfigParser()
    config.read(config_path)
    print_log(f"Loading base config: {config_path.stem}{config_path.suffix}")
    return config

def generate_config(config_name: Path | str):
    """
    Generate the config file for RAG chatbot.

    Args:
        config_name: Config filename.
    """

    def normalize_column_name(name: str) -> str:
        name = re.sub(r"[\s\-]+", "_", str(name).strip())
        name = re.sub(r"[^a-zA-Z0-9_]", "", name)
        name = re.sub(r"_+", "_", name)
        return name.strip("_")

    base_config = load_config(config_name)

    output_path = f"{base_config['RAG']['rag_name']}_config.ini"
    output_path = Path(PROJECT_DIR / output_path)

    base_config["DATASET"]["path"] = f"{PROJECT_DIR}/{base_config['DATASET'].get('path')}"
    base_config["DATASET"]["max_rows"] = base_config['DATASET'].get("max_rows", "None")

    tmp_col = base_config['DATASET'].get("main_columns", "None")
    tmp_col = ", ".join([normalize_column_name(col) for col in config_sep2list(tmp_col)])
    base_config["DATASET"]["main_columns"] = tmp_col.strip()

    tmp_col = base_config['DATASET'].get("drop_columns", "None")
    tmp_col = ", ".join([normalize_column_name(col) for col in config_sep2list(tmp_col)])
    base_config["DATASET"]["drop_columns"] = tmp_col.strip()

    base_config["LLM"] = {
        "model_name": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
        "n_ctx": 8096,
        "max_history_messages": 2,
        "system_prompt": f"""You are {base_config['RAG']['rag_name']}, a friendly and knowledgeable assistant.

    You can have normal conversations with the user without using any tools.

    Examples of messages that should NOT use any tool:
    - Hi
    - Hello
    - Who are you?
    - Thank you
    - What can you do?
    - Tell me a joke

    For these messages, reply naturally as a friendly chatbot.
    --------------------------------------------------
    Use tools ONLY when the user is requesting information that requires searching the local database. Choose the correct tool below:

    • search_by_columns
    Use when the user asking some information in certain column.

    • search_by_filter
    Use when you need to filters the dataset based on conditions (like greater than, less than, or equals) and returns a specific number of rows.

    • basic_aggregation
    Calculates math like MIN, MAX, AVG, SUM, COUNT, or COUNT_DISTINCT on a specific numeric column. Support GROUP BY and HAVING if needed.

    Tool-calling rules:
    - Only use tools when the user's request requires searching into database.
    - It is completely acceptable to answer without calling any tool.
    - Never guess missing tool arguments.
    - Only pass information explicitly provided by the user.
    - Leave unspecified optional arguments null.
    - After receiving tool results, answer only using the returned data.
    - If the tool returns no results, politely inform the user that no matching data were found.
    """,
    }

    with open(output_path, 'w') as configfile:
        base_config.write(configfile)
    print_log(f"Generating config file: {output_path.stem}{output_path.suffix}")

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="path to base_config.cfg", required=True)

    args = parser.parse_args()
    generate_config(Path(args.name))