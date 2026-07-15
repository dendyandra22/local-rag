# from ollama import chat
# # from ollama import ChatResponse
# # import pandas as pd
# # import sqlite3
# #
# # from .sqldb import SQLDB
# # from util.logger import print_log
# import re
# from pathlib import Path
# from nlu_module.nlu_component import NLUComponent
# from database_module.vectordb import VectorDB
# from database_module.sqldb import SQLDB
# from database_module.tabular_data import read_tabular_dataset
#
#
# BASE_DIR = Path(__file__).resolve().parent.parent
# DATA_DIR = BASE_DIR / "data"
# MODEL_DIR = BASE_DIR / "model"
#
#
# class Response:
#     def __init__(self, db_name, timestamp, inference_type, rebuild_rag):
#         self.db_name = db_name
#
#         if self._check_exist_rag(db_name, timestamp):
#             if rebuild_rag:
#                 print(f"rebuild existing rag {db_name}f")
#             else:
#                 print(f"use existing rag {db_name}")
#
#         self.nlu_model = NLUComponent(timestamp, retrain_model=rebuild_rag)
#         self.inference_type = inference_type
#
#         if rebuild_rag:
#             col_list = ['mal_id', 'title', 'title_english', 'score', 'authors', 'genres', 'themes', 'synopsis']
#             self._build_database(source_path='dataset/manga_dataset.csv', col_list=col_list, max_rows=None)
#
#         else:
#             self.sqm = SQLDB(db_name,
#                              rebuild_db=rebuild_rag,
#                              )
#             self.vectordb = VectorDB(db_name=db_name,
#                                      rebuild_db=rebuild_rag,
#                                     )
#
#
#     def _build_database(self, source_path: str, col_list: list[str], max_rows: int | None = None):
#         df = read_tabular_dataset(source_path)
#         max_rows = max_rows if isinstance(max_rows, int) else df.shape[0]
#         df.dropna(subset=['title', 'score', 'authors', 'genres', 'themes', 'synopsis'], inplace=True)
#
#         self.sqm = SQLDB(db_name=self.db_name,
#                          rebuild_db=True,
#                          df=df
#                          )
#
#         df = df.loc[:max_rows, col_list]
#         df['authors'] = df['authors'].apply(lambda x: x.replace('|', '\n'))
#         df['genres'] = df['genres'].apply(lambda x: x.replace('|', '\n'))
#         df['themes'] = df['themes'].apply(lambda x: x.replace('|', '\n'))
#
#         self.vectordb = VectorDB(db_name=self.db_name,
#                                  rebuild_db=True,
#                                  df=df
#                                  )
#
#
#     @staticmethod
#     def _format_chat_history(chat_history):
#         if not chat_history:
#             return ''
#
#         recent_history = chat_history[-8:]
#         history_lines = []
#         for item in recent_history:
#             role = item.get('role', 'user')
#             content = item.get('content', '')
#             if content:
#                 history_lines.append(f'{role}: {content}')
#
#         if not history_lines:
#             return ''
#
#         return 'Conversation history:\n' + '\n'.join(history_lines)
#
#     @staticmethod
#     def _clean_text(text):
#         text = text.replace('|', ', ')
#         text = re.sub(r"\(Source:.*?\)", " ", text)
#         text = re.sub(r"\s+", " ", text).strip()
#
#         return text
#
#     def generate_chat(self, context, sys_prompt, stream=False, chat_history=None):
#         response = None
#
#         if self.inference_type == 'ollama':
#             history_context = self._format_chat_history(chat_history)
#             # content = f'''
#             #     history context:
#             #     {history_context}
#             #
#             #     user question:
#             #     {message}
#             #     '''
#
#
#             template_message = [
#                 {
#                     'role': 'system',
#                     'content': sys_prompt
#                 },
#                 {
#
#                     'role': 'user',
#                     'content': context,
#                 },
#             ]
#
#             print('TEMPLATE MESSAGE', template_message)
#
#             # if context != '':
#             #     content = f'''
#             #     {context}
#             #     {history_context}
#             #     user question:
#             #     {message}
#             #     '''
#             #     template_message = [
#             #         {
#             #             'role': 'system',
#             #             'content': '''
#             #                             You are MangaBot, a friendly and knowledgeable manga assistant.
#             #
#             #                             Your personality:
#             #                             - Be friendly and conversational.
#             #                             - Answer naturally, like chatting with another manga fan.
#             #                             - Give complete answers instead of one short sentence.
#             #                             - When appropriate, add a little extra helpful information.
#             #                             - Be enthusiastic about manga, but do not invent facts.
#             #                             - If the user asks casual questions (such as "Who are you?" or "How are you?"), respond naturally instead of redirecting to manga.
#             #                             - Keep responses around 2–5 sentences unless the user requests more detail.
#             #                             '''
#             #         },
#             #         {
#             #
#             #             'role': 'user',
#             #             'content': content,
#             #         },
#             #     ]
#
#             response = chat(model='gemma3:4b',
#                             messages=template_message,
#                             stream=stream,
#                             # options={
#                             #     # 'temperature': 0.7,
#                             #     # Bumps creativity and conversational flow (default is often too low)
#                             #     # 'repeat_penalty': 1.0,  # Prevents it from getting stuck looping phrases
#                             #     'num_predict': 256  # Encourages a decent length response
#                             # }
#                             )
#
#         return response
#
#     def get_from_sql(self, query_filter, use_fts, return_context, limit: int = 5):
#
#         if use_fts:
#             query = f'''
#             SELECT * FROM {self.db_name}_fts
#             WHERE {self.db_name}_fts MATCH ''' + query_filter
#             # context = ''
#
#
#         else:
#             query = f'''SELECT * FROM {self.db_name} ''' + query_filter
#             # context = ''
#
#         print('XXX SQL QUERY', query)
#         result = self.sqm.search_sql(query, limit, return_context)
#         return result
#         # conn = self.sqm.sql_connect()
#         # try:
#         #     cursor = conn.cursor()
#         #     cursor.execute(query)
#         #     all_rows = cursor.fetchmany(limit)
#         #
#         #     col_list = [desc[0] for desc in cursor.description]
#         # finally:
#         #     conn.close()
#         #
#         # if return_context:
#         #     for row in all_rows:
#         #         context = context + ' '.join(f'{col}: {val}\n' for col, val in zip(col_list, row))
#         #
#         #     return context
#         #
#         # data = [{k: v for k, v in zip(col_list, row)} for row in all_rows]
#         # return data
#
#     def get_from_vectordb(self, message, k, return_context):
#         message = self._clean_text(message)
#         mal_id, distances = self.vectordb.search_vector(message, k)
#         mal_id = [str(ids) for ids in mal_id[:k]]
#         print('response mal_id', mal_id)
#         print('distances mal_id', distances)
#         context = ''
#         if len(mal_id) > 0:
#             placeholders = ",".join(mal_id)
#             query_filter = f'WHERE mal_id IN ({placeholders})'
#             context = self.get_from_sql(query_filter, use_fts=False, limit=k, return_context=return_context)
#
#         return context
