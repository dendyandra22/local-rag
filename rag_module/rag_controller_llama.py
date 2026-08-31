import re
from pathlib import Path
import pandas as pd
from ollama import chat
import time
import configparser
from llama_cpp import Llama
import json
import ast

from nlu_module.nlu_component import NLUComponent
# from rag_module.response import Response
from util.logger import print_log
from database_module.vectordb import VectorDB
from database_module.sqldb import SQLDB
# from rag_module.manga_tool import *
from rag_module.general_tool import sql_get_by_columns, sql_get_by_filter
from rag_module.rag_staging import *

SUPPORTED_INFERENCE = ["llama"]
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".xls", ".xlsx"}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECT_DIR = BASE_DIR / "dataset" # temp project dir is dataset

class RAGModel:
    def __init__(self,
                 rag_name: str,
                 # nlu_name: str | None,
                 # inference_type: str | None,
                 ):
        self.rag_name = rag_name
        self._config = self.load_config(PROJECT_DIR / f"{rag_name}.ini")
        self.nlu_name = self._config["NLU"]["nlu_name"]
        self.inference_type = self._config["LLM"]["inference_type"]
        if self.inference_type != "ollama":
            self.llm = self._load_local_llm()
        self.sql_db = None
        self.vec_db = None
        self.connect_rag_db()
        self.nlu = NLUComponent(self.nlu_name, retrain_model=False) if self.nlu_name else None

    @staticmethod
    def _load_local_llm():
        llm = Llama(
            model_path="model/Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            seed=42,  # Uncomment to set a specific seed
            n_ctx=2048, # Uncomment to increase the context window
            verbose=False
        )

        return llm

    def connect_rag_db(self):
        sql, vec = None, None
        try:
            sql = SQLDB(self.rag_name)
            sql.connect_db()
        except Exception as e:
            print_log(str(e))
            sql = None

        try:
            vec = VectorDB(self.rag_name)
            vec.connect_db()
        except Exception as e:
            print_log(str(e))
            vec = None

        if sql and vec:
            print_log("successfully connected to RAG database")

        # return sql, vec
        self.sql_db = sql
        self.vec_db = vec


    @staticmethod
    def load_config(config_name):
        config_path = Path(config_name)
        if not config_path.is_file():
            raise FileNotFoundError(f"{config_path.stem}{config_path.suffix} not found in dataset directory.")

        config = configparser.ConfigParser()
        config.read(config_path)
        print_log(f"Loading RAG config: {config_path.stem}{config_path.suffix}")

        return config

    @staticmethod
    def _config_sep2list(value):
        if value is None:
            return []

        result = [
            f.strip()
            for f in value.split(", ")
        ]

        return result

    @staticmethod
    def read_tabular_dataset(source_path: str | Path) -> pd.DataFrame:
        path = Path(source_path)
        extension = path.suffix.lower()

        if extension == ".csv":
            return pd.read_csv(path)

        if extension in {".xls", ".xlsx"}:
            return pd.read_excel(path)

        supported = ", ".join(sorted(SUPPORTED_DATASET_EXTENSIONS))
        raise ValueError(f"Unsupported dataset file type. Use one of: {supported}")


class RAGAction(RAGModel):
    def __init__(self,
                 rag_name: str | None = None,
                 ):
        super().__init__(rag_name)

    def _verify_rag_available(self):
        """Raises an error if the model isn't configured."""
        # if self.nlu is None or self.sql_db is None or self.vec_db is None:
        #     raise AttributeError('RAG model components are not initialized.')

        if self.nlu is None:
            raise AttributeError('RAG NLU components are not initialized.')
        if self.sql_db is None:
            raise AttributeError('RAG SQL DB components are not initialized.')
        if self.vec_db is None:
            raise AttributeError('RAG VEC DB components are not initialized.')


    def create_rag_from_dataset(self):
        """
        Creates RAG database from provided dataset in project directory "/dataset".
        All setting are done in respective config files in the project directory.
        """
        print_log("create rag from dataset")
        # get config data
        retrain_nlu = bool(self._config["NLU"].get("retrain_nlu"))
        df_path = self._config["DATASET"].get("path")
        max_rows = self._config["DATASET"].get("max_rows")
        drop_na_cols = self._config_sep2list(self._config["DATASET"].get("drop_na_fields"))
        index_column = self._config["DATASET"].get("index_column")
        clean_fields = self._config_sep2list(self._config["EMBEDDING"].get("clean_fields"))
        embed_fields = self._config_sep2list(self._config["EMBEDDING"].get("fields"))


        df = self.read_tabular_dataset(df_path)

        max_rows = int(max_rows) if max_rows is not None else df.shape[0]
        df = df.loc[:max_rows]
        print_log(f"{max_rows} rows data selected")
        
        if isinstance(drop_na_cols, list) and len(drop_na_cols) > 0:
            df.dropna(subset=drop_na_cols, inplace=True)
            print_log("dropping unused columns")

        # if isinstance(clean_fields, list) and len(clean_fields) > 0:
        #     for col in clean_fields:
        #         df[col] = df[col].apply(lambda x: self._clean_text(x))
        #         print_log(f"{col} column cleaned")

        self.sql_db = SQLDB(self.rag_name).create_db(df)

        if isinstance(embed_fields, list) and len(embed_fields) > 0:
            df = df.loc[:max_rows, embed_fields]

        if index_column is None:
            df.reset_index(drop=False, inplace=True)
            index_column = 'index'
        self.vec_db = VectorDB(self.rag_name).create_db(df, index_col=index_column)
        
        if retrain_nlu:
            self.nlu = NLUComponent(self.nlu_name, retrain_model=retrain_nlu)


    # general tool calling
    def search_by_columns(self, args: dict, searchable_columns: list[str]):
        """
            Finds data matching a combination of specific columns and value mentioned by users.

            CRITICAL: Only provide arguments that the user EXPLICITLY mentioned in their prompt.
            DO NOT guess, hallucinate, or assume these values if they aren't provided by the user.
        """
        tool_output = sql_get_by_columns(self.sql_db,
                                        args=args,
                                        searchable_columns=searchable_columns,
                                        operator="AND"
                                        )
        return tool_output

    def search_by_filter(self, filters: list[dict], searchable_columns: list[str], limit: int = 10):
        """
        Filters the dataset based on conditions (like greater than, less than, or equals) and returns a specific number of rows.
        """
        print("def searchable col:", searchable_columns)
        print("filters\n", filters)
        print("limit\n", limit)

        tool_output = sql_get_by_filter(self.sql_db, filter_list=filters, limit=limit, searchable_columns=searchable_columns)
        print("search_by_filter output\n", tool_output)


        return None



    def response_handler_with_tool(self, messages: str, stream: bool=False, verbose: bool=False, chat_history: list[dict]|None=None, chat_history_limit: int = 5):
        self._verify_rag_available()
        context = ''
        sys_prompt = self._config["LLM"].get("system_prompt")
        if sys_prompt is None: raise ValueError("no system prompt in config file! please provide prompt")

        template_messages = [
            {
                'role': 'system',
                'content': sys_prompt,
            }]
        if chat_history is not None and len(chat_history) > 0:
            template_messages.extend(chat_history[-chat_history_limit:])

        template_messages.append(
            {
                'role': 'user',
                'content': messages,
            }
        )
        print_log("Processing user message")
        print(template_messages)
        # Structure the tool definitions using OpenAI format
        searchable_columns = self._config_sep2list(self._config["DATASET"].get("drop_na_fields"))
        properties = {}
        for col in searchable_columns:
            properties[col] = {
                "type": "string",
                "description": f"The exact value for {col} mentioned by the user. Leave null if not mentioned."
            }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_by_columns",
                    "description": "Finds data matching specific columns. ONLY provide arguments explicitly mentioned by the user.",
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_by_filter",
                    "description": "Filters the dataset based on conditions (like greater than, less than, or equals) and returns a specific number of rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filters": {
                                "type": "array",
                                "description": "A list of conditions to apply to the data.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column": {
                                            "type": "string",
                                            "description": "The database column to filter.",
                                            "enum": searchable_columns  # Force the LLM to pick a valid column
                                        },
                                        "operator": {
                                            "type": "string",
                                            "description": "The comparison operator.",
                                            "enum": ["=", ">", "<", ">=", "<="]  # Force valid SQL operators
                                        },
                                        "value": {
                                            "type": "string",
                                            "description": "The value to compare against. (e.g., 'games', '7.5')"
                                        }
                                    },
                                    "required": ["column", "operator", "value"]
                                }
                            },
                            "limit": {
                                "type": "integer",
                                "description": "The maximum number of rows to return. Default to 10 if the user doesn't specify.",
                                "default": 10
                            }
                        },
                        "required": ["filters"]
                    }
                }
            }
        ]

        first_stage_output = hf_first_stage(self.llm,
                                            template_messages=template_messages,
                                            tools=tools,
                                            tool_choice="auto"
                                            )
        print_log("First LLM stage")
        choices = first_stage_output["choices"][0]["message"]
        print_log(choices)
        template_messages.append(choices)

        # catch tool calls in dict keys
        tool_name, tool_args = None, None
        if "tool_calls" in choices and choices["tool_calls"]:
            print_log("Tool Call LLM stage")
            for tool_call in choices["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_args = tool_args["args"]
                # tool_call_id = tool_call["id"]

        # catch tool calls if its written on content
        elif choices.get("content") and "<tool_call>" in choices["content"]:
            # extract the JSON block between the tags using regex
            print_log("Extracting tool call from content")
            match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", choices["content"], re.DOTALL)
            if match:
                raw_json_str = match.group(1)
                # print("raw_json_str\n",raw_json_str)
                try:
                    tool_data = json.loads(raw_json_str)
                except json.decoder.JSONDecodeError:
                    print_log("JSONDecodeError raised! fixing raw json string")
                    raw_json_str += "}"
                    tool_data = json.loads(raw_json_str)
                tool_name = tool_data["name"]
                tool_args = tool_data["arguments"]
                if "args" in tool_args:
                    tool_args = tool_args["args"]

        if tool_name and tool_args:
            tool_output = None
            print(f"-> QWEN triggered {tool_name} call with arguments: {tool_args} type args: {type(tool_args)}")

            if tool_name == "search_by_columns":
                tool_output = self.search_by_columns(tool_args, searchable_columns=searchable_columns)

            elif tool_name == "search_by_filter":
                filters = tool_args.get("filters", [])
                limit = tool_args.get("limit", 10)
                tool_output = self.search_by_filter(filters=filters,
                                                         limit=limit,
                                                         searchable_columns=searchable_columns
                                                         )

            # Append the tool result back to the message history
            if tool_output is not None:
                template_messages.append({
                    "role": "tool",
                    # "tool_call_id": tool_call_id,
                    "tool_name": tool_name,
                    "content": tool_output
                })
            raise ValueError("END TEST")
            print_log("Second LLM stage")
            print(template_messages)

            # stream final response
            def _response_generator():
                result = hf_second_stage(self.llm, template_messages=template_messages, tools=tools)
                for chunk in result:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]

            return _response_generator()

        # if llm doesnt need to call tool, return response with pseudo stream
        else:
            print_log("First LLM stage - pseudo stream")
            def _response_generator():
                for i in range(0, len(choices["content"]), 20):
                    yield choices["content"][i:i + 20]
                    time.sleep(0.2)

            return _response_generator()

