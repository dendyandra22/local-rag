from ollama import chat
# from ollama import ChatResponse
# import pandas as pd
# import sqlite3
#
# from .sqldb import SQLDB
# from util.logger import print_log
from nlu_module.nlu_component import NLUComponent
from rag_module.vectordb import VectorDB
from rag_module.sqldb import SQLDB


class Response:
    def __init__(self, db_name, timestamp, inference_type, retrain_model):
        self.db_name = db_name
        self.nlu_model = NLUComponent(timestamp, retrain_model=retrain_model)
        self.inference_type = inference_type
        self.sqm = SQLDB(db_name,
                         # source_path='dataset/manga_dataset.csv',
                         # rebuild_db=0,
                         # max_rows=5000
                         )
        self.vectordb = VectorDB(db_name=db_name)

    def _format_chat_history(self, chat_history):
        if not chat_history:
            return ''

        recent_history = chat_history[-8:]
        history_lines = []
        for item in recent_history:
            role = item.get('role', 'user')
            content = item.get('content', '')
            if content:
                history_lines.append(f'{role}: {content}')

        if not history_lines:
            return ''

        return 'Conversation history:\n' + '\n'.join(history_lines)

    def generate_chat(self, message, context, sys_prompt, stream=False, chat_history=None):
        response = None

        if self.inference_type == 'ollama':
            history_context = self._format_chat_history(chat_history)
            content = f'''
                {history_context}
                user question:
                {message}
                '''
            template_message = [
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {

                    'role': 'user',
                    'content': content,
                },
            ]

            if context != '':
                content = f'''
                {context}
                {history_context}
                user question:
                {message}
                '''
                template_message = [
                    {
                        'role': 'system',
                        'content': '''
                                        You are a manga assistant.

                                        Use the information provided in the Context section if needed.

                                        Rules:
                                        1. Answer only from the context.
                                        2. Do not use your own knowledge.
                                        3. Be concise and factual.
                                        4. If multiple manga provided, explain each manga clearly based on information in context.
                                        5. If no context given, try to answer by your own knowledge.
                                        6. Use Conversation history only to understand references in the current user question.
                                        '''
                    },
                    {

                        'role': 'user',
                        'content': content,
                    },
                ]

            response = chat(model='gemma3:4b', messages=template_message, stream=stream)

        return response

    def get_from_sql(self, query_filter, use_fts, return_context, limit: int = 5):

        if use_fts:
            query = f'''
            SELECT * FROM {self.db_name}_fts
            WHERE {self.db_name}_fts MATCH ''' + query_filter
            context = ''


        else:
            query = f'''SELECT * FROM {self.db_name} ''' + query_filter
            context = ''

        conn = self.sqm.sql_connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            all_rows = cursor.fetchmany(limit)

            col_list = [desc[0] for desc in cursor.description]
        finally:
            conn.close()

        if return_context:
            for row in all_rows:
                context = context + ' '.join(f'{col}: {val}\n' for col, val in zip(col_list, row))

            return context

        data = [{k: v for k, v in zip(col_list, row)} for row in all_rows]
        return data

    def get_from_vectordb(self, message, k, return_context):
        mal_id = self.vectordb.search_vector(message)
        mal_id = [str(ids) for ids in mal_id[:k]]
        context = ''
        if len(mal_id) > 0:
            placeholders = ",".join(mal_id)
            query_filter = f'WHERE mal_id IN ({placeholders})'
            context = self.get_from_sql(query_filter, use_fts=False, limit=k, return_context=return_context)

        return context
