import re
from pathlib import Path
import pandas as pd
from ollama import chat
import time
from llama_cpp import Llama
import json

from util.logger import print_log
from database_module.vectordb import VectorDB
from database_module.sqldb import SQLDB
from rag_module.general_tool import *

# general tool calling function

def ollama_first_stage(model_name, template_messages: list[dict], tools: list = None, verbose: bool = False):
    tool_detail = {
        "tool_name": None,
        "args": None,
        "response": None,
    }

    response = chat(
        model=model_name,
        chat_format="chatml-function-calling",
        messages=template_messages,
        stream=False,
        tools=tools,
    )
    tool_detail["response"] = response
    if response.message.tool_calls is not None:
        for tool in response.message.tool_calls:
            tool_detail["tool_name"] = tool.function.name
            tool_detail["args"] = tool.function.arguments
            if verbose: print_log(f"""-> Gemma 3 triggered {tool_detail["tool_name"]} call with arguments: {tool_detail["args"]}""")

    return tool_detail

def hf_first_stage(model, template_messages: list[dict], tools: list = None, tool_choice: str = None, verbose: bool = False):
    response = model.create_chat_completion(
        messages=template_messages,
        stream=False,
        max_tokens=None,
        tools=tools,
        tool_choice=tool_choice,
    )
    if verbose: print_log("RAG STAGING 1 HF!!!")

    return response

def hf_second_stage(model, template_messages: list[dict], tools: list = None, verbose: bool = False):
    response = model.create_chat_completion(
        messages=template_messages,
        stream=True,
        max_tokens=None,
        tools=tools,
    )
    if verbose: print_log("RAG STAGING 2 HF!!!")

    return response


# def ollama_tool_calling_handler(
#     response,
#     template_messages,
#     tools: list = None,
#     verbose: bool = True,
# ):
#     """Handle tool calling for Ollama inference stage"""
#
#     template_messages.append(response.message)
#     for tool in response.message.tool_calls:
#         tool_name = tool.function.name
#         tool_output = None
#         args = tool.function.arguments
#         print(f"-> Gemma 3 triggered {tool_name} call with arguments: {args}")
#
#         if tool_name == "search_by_columns":
#             # Execute the local function
#             tool_output = search_by_columns(args)
#
#         # elif tool_name == "search_by_synopsis":
#         #     # Execute the local function
#         #     tool_output = search_by_synopsis(self.vec_db, query=args.get("query"))
#         #
#         #
#         # elif tool_name == "recommend_by_title":
#         #     # Execute the local function
#         #     tool_output = recommend_by_title(self.sql_db, title=args.get("title"))
#
#         if tool_output is not None:
#             # Feed the function's result back into the chat history
#             print("TOOL OUTPUT", tool_output)
#             template_messages.append({
#                 "role": "tool",
#                 "tool_name": tool_name,
#                 "content": tool_output
#             })
#
#     print()
#     print("RH-AFTER TOOL CALLS")
#     print(template_messages)
#     # if stream:
#     if verbose: print('RH-STREAM_TOOL')
#
#     def _response_generator():
#         result = chat(
#             model="lukaspetrik/gemma3-tools:4b",
#             messages=template_messages,
#             stream=stream,
#         )
#         for chunk in result:
#             yield chunk['message']['content']
#
#     return _response_generator()
