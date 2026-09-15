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

from util.logger import print_log
from database_module.vectordb import VectorDB
from database_module.sqldb import SQLDB
from rag_module.general_tool import sql_get_by_columns, sql_get_by_filter
from rag_module.rag_staging import *

SUPPORTED_INFERENCE = ["llama"]
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".xls", ".xlsx"}

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
PROJECT_DIR = BASE_DIR / "dataset" # temp project dir is dataset

class RAGModel:
    def __init__(self,
                 rag_name: str,
                 ):
        self.rag_name = rag_name
        self._config = self.load_config(PROJECT_DIR / f"{rag_name}_config.ini")
        self.llm = self._load_local_llm()
        self.sql_db = None
        self.connect_rag_db()

    def _load_local_llm(self):
        llm = Llama(
            model_path=f"{MODEL_DIR}/{self._config['LLM'].get('model_name')}",
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            seed=42,
            n_ctx=int(self._config["LLM"].get("n_ctx", "8096")),
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

        # try:
        #     vec = VectorDB(self.rag_name)
        #     vec.connect_db()
        # except Exception as e:
        #     print_log(str(e))
        #     vec = None

        if sql and vec:
            print_log("successfully connected to RAG database")

        # return sql, vec
        self.sql_db = sql
        # self.vec_db = vec


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
                 verbose: bool = False,
                 ):
        super().__init__(rag_name)
        self.verbose = verbose

    def _verify_rag_available(self):
        """Raises an error if the model isn't configured."""
        if self.sql_db is None:
            raise AttributeError('RAG SQL DB components are not initialized.')
        # if self.vec_db is None:
        #     raise AttributeError('RAG VEC DB components are not initialized.')


    def create_rag_from_dataset(self):
        """
        Creates RAG database from provided dataset in project directory "/dataset".
        All setting are done in respective config files in the project directory.
        """

        def normalize_column_name(name: str) -> str:
            name = re.sub(r"[\s\-]+", "_", str(name).strip())
            name = re.sub(r"[^a-zA-Z0-9_]", "", name)
            name = re.sub(r"_+", "_", name)
            return name.strip("_")

        print_log("create rag from dataset")

        # get config data
        df_path = self._config["DATASET"].get("path")
        max_rows = self._config["DATASET"].get("max_rows")
        try:
            max_rows = int(max_rows)
        except ValueError:
            max_rows = None
        drop_na_cols = self._config_sep2list(self._config["DATASET"].get("main_columns"))
        drop_columns = self._config_sep2list(self._config["DATASET"].get("drop_columns"))
        # index_column = self._config["DATASET"].get("index_column")
        # clean_fields = self._config_sep2list(self._config["EMBEDDING"].get("clean_fields"))
        # embed_fields = self._config_sep2list(self._config["EMBEDDING"].get("fields"))


        df = self.read_tabular_dataset(df_path)

        max_rows = int(max_rows) if max_rows is not None else df.shape[0]
        df = df.loc[:max_rows]
        print_log(f"{max_rows} rows data selected")

        # handle whitespace in column name
        df.columns = [normalize_column_name(col) for col in df.columns]

        if isinstance(drop_na_cols, list) and len(drop_na_cols) > 0:
            df.dropna(subset=drop_na_cols, inplace=True)
            print_log("Dropping null values")

        if isinstance(drop_columns, list) and len(drop_columns) > 0:
            df.drop(columns=drop_columns, inplace=True)
            print_log("Dropping unused columns")

        # if isinstance(clean_fields, list) and len(clean_fields) > 0:
        #     for col in clean_fields:
        #         df[col] = df[col].apply(lambda x: self._clean_text(x))
        #         print_log(f"{col} column cleaned")

        self.sql_db = SQLDB(self.rag_name).create_db(df)

        # if isinstance(embed_fields, list) and len(embed_fields) > 0:
        #     df = df.loc[:max_rows, embed_fields]
        #
        # if index_column is None:
        #     df.reset_index(drop=False, inplace=True)
        #     index_column = 'index'
        # self.vec_db = VectorDB(self.rag_name).create_db(df, index_col=index_column)
        #
        # if retrain_nlu:
        #     self.nlu = NLUComponent(self.nlu_name, retrain_model=retrain_nlu)


    def response_handler_with_tool(self, messages: str, chat_history: list[dict]|None=None, chat_history_limit: int = 5):
        print("verbsoe stat:", self.verbose)
        self._verify_rag_available()
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
        # Structure the tool definitions using OpenAI format
        searchable_columns = self._config_sep2list(self._config["DATASET"].get("main_columns"))
        tool_schema = generate_toolcall_schema(searchable_columns)

        if self.verbose: print_log("First LLM stage")
        first_stage_output = hf_first_stage(self.llm,
                                            template_messages=template_messages,
                                            tools=tool_schema,
                                            tool_choice="auto"
                                            )

        choices = first_stage_output["choices"][0]["message"]
        if self.verbose: print_log(f"First LLM stage output:\n {choices}")
        template_messages.append(choices)

        # catch tool calls in dict keys
        tool_name, tool_args = None, None
        if "tool_calls" in choices and choices["tool_calls"]:
            if self.verbose: print_log("Tool Call LLM stage")
            for tool_call in choices["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_args = tool_args["args"]

        # catch tool calls if it's written on content
        elif choices.get("content") and "<tool_call>" in choices["content"]:
            # extract the JSON block between the tags using regex
            if self.verbose: print_log("Extracting tool call from content")
            match = re.search(r"<tool_call>\s*({.*?})\s*</tool_call>", choices["content"], re.DOTALL)
            if match:
                raw_json_str = match.group(1)
                try:
                    tool_data = json.loads(raw_json_str)
                except json.decoder.JSONDecodeError:
                    if self.verbose: print_log("JSONDecodeError raised! fixing raw json string")
                    raw_json_str += "}"
                    tool_data = json.loads(raw_json_str)
                tool_name = tool_data["name"]
                tool_args = tool_data["arguments"]
                if "args" in tool_args:
                    tool_args = tool_args["args"]

        if tool_name and tool_args:
            tool_output = None
            # print(f"-> QWEN triggered {tool_name} call with arguments: {tool_args} type args: {type(tool_args)}")

            if tool_name == "search_by_columns":
                tool_output = sql_get_by_columns(self.sql_db,
                                        args=tool_args,
                                        searchable_columns=searchable_columns,
                                        operator="AND",
                                        verbose=self.verbose
                                        )

            elif tool_name == "search_by_filter":
                filters = tool_args.get("filters", [])
                limit = tool_args.get("limit", 10)
                tool_output = sql_get_by_filter(self.sql_db,
                                                filter_list=filters,
                                                limit=limit,
                                                searchable_columns=searchable_columns,
                                                verbose=self.verbose
                                                )

            elif tool_name == "basic_aggregation":
                aggregation_list = tool_args.get("aggregation_list", [])
                filters = tool_args.get("filters", [])
                group_by = tool_args.get("group_by", [])
                having = tool_args.get("having", {})
                tool_output = sql_get_basic_aggregation(self.sql_db, aggregation_list=aggregation_list,
                                                        filter_list=filters,
                                                        group_by=group_by,
                                                        having=having,
                                                        limit=None,
                                                        verbose=self.verbose
                                                        )

            # Append the tool result back to the message history
            if tool_output is not None:
                template_messages.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": tool_output
                })

            if self.verbose: print_log("Second LLM stage")

            # stream final response
            def _response_generator():
                result = hf_second_stage(self.llm, template_messages=template_messages, tools=tool_schema)
                for chunk in result:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        yield delta["content"]

            return _response_generator()

        # if llm doesnt need to call tool, return response with pseudo stream
        else:
            if self.verbose: print_log("First LLM stage - pseudo stream")
            def _response_generator():
                for i in range(0, len(choices["content"]), 20):
                    yield choices["content"][i:i + 20]
                    time.sleep(0.2)

            return _response_generator()

