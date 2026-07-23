import re
from pathlib import Path
import pandas as pd
from ollama import chat
import time

from nlu_module.nlu_component import NLUComponent
# from rag_module.response import Response
from util.logger import print_log
from database_module.vectordb import VectorDB
from database_module.sqldb import SQLDB
from rag_module.manga_tool import *

SUPPORTED_INFERENCE = ["ollama", 'huggingface']
SUPPORTED_DATASET_EXTENSIONS = {".csv", ".xls", ".xlsx"}

class RAGModel:
    def __init__(self,
                 rag_name: str,
                 nlu_name: str | None,
                 inference_type: str | None,
                 ):
        self.rag_name = rag_name
        self.nlu_name = nlu_name
        self.inference_type = inference_type if inference_type in SUPPORTED_INFERENCE else 'ollama'
        # self.llm = None
        self.sql_db = None
        self.vec_db = None
        self.connect_rag_db()
        self.nlu = NLUComponent(self.nlu_name, retrain_model=False) if nlu_name else None

    def connect_rag_db(self):
        sql, vec = None, None
        try:
            sql = SQLDB(self.rag_name)
            vec = VectorDB(self.rag_name)
            sql.connect_db()
            vec.connect_db()

        except Exception as e:
            print_log(str(e))

        else:
            print_log("successfully connected to RAG database")

        # return sql, vec
        self.sql_db = sql
        self.vec_db = vec

    def rebuild_rag(self, df: pd.DataFrame, retrain_nlu: bool = True):
        sql, vec, nlu = None, None, None
        try:
            sql = SQLDB(self.rag_name).create_db(df)
            vec = VectorDB(self.rag_name).create_db(df)
            if retrain_nlu:
                nlu = NLUComponent(self.nlu_name, retrain_model=retrain_nlu)
        except Exception as e:
            print_log(str(e))

        return sql, vec, nlu
        # self.sql_db = sql
        # self.vec_db = vec
        # self.nlu = nlu

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
    
    def ollama_generate_chat(self, context, sys_prompt, stream=False):
        response = None

        if self.inference_type == 'ollama':

            template_message = [
                {
                    'role': 'system',
                    'content': sys_prompt
                },
                {

                    'role': 'user',
                    'content': context,
                },
            ]

            # print('TEMPLATE MESSAGE', template_message)

            response = chat(model='gemma3:4b',
                            messages=template_message,
                            stream=stream,
                            # options={
                            #     # 'temperature': 0.7,
                            #     # Bumps creativity and conversational flow (default is often too low)
                            #     # 'repeat_penalty': 1.0,  # Prevents it from getting stuck looping phrases
                            #     'num_predict': 256  # Encourages a decent length response
                            # }
                            )

        return response






class RAGAction(RAGModel):
    def __init__(self,
                 rag_name: str | None = None,
                 nlu_model_name: str | None = None,
                 inference_type: str | None = None,
                 ):
        super().__init__(rag_name, nlu_model_name, inference_type)

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


    @staticmethod
    def _clean_text(text):
        text = re.sub(r"\(Source:.*?\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def create_rag_from_dataset(self, source_path: str, 
                                col_list: list[str],
                                max_rows: int | None = None,
                                drop_cols: list[str] | None = None,
                                retrain_nlu: bool = True,
                                ):
        print_log("create rag from dataset")
        df = self.read_tabular_dataset(source_path)
        max_rows = max_rows if isinstance(max_rows, int) else df.shape[0]
        print_log(f"{max_rows} rows data selected")
        
        if isinstance(drop_cols, list) and len(drop_cols) > 0:
            df.dropna(subset=drop_cols, inplace=True)
            print_log("dropping unused columns")
            # df.dropna(subset=['title', 'score', 'authors', 'genres', 'themes', 'synopsis'], inplace=True)

        df['synopsis'] = df['synopsis'].apply(lambda x: self._clean_text(x))
       
        self.sql_db = SQLDB(self.rag_name).create_db(df)

        df = df.loc[:max_rows, col_list]
        df['authors'] = df['authors'].apply(lambda x: x.replace('|', '\n'))
        df['genres'] = df['genres'].apply(lambda x: x.replace('|', '\n'))
        df['themes'] = df['themes'].apply(lambda x: x.replace('|', '\n'))

        df_vec = df[["mal_id","title","title_english","synopsis"]]
        
        self.vec_db = VectorDB(self.rag_name).create_db(df_vec)
        
        if retrain_nlu:
            self.nlu = NLUComponent(self.nlu_name, retrain_model=retrain_nlu)
            

    def _metadata_lookup_query(self, message, sort_rating=True):
        self._verify_rag_available()
        query_filter = ''
        message = re.sub(
            r'\b(?:by\s+author|with\s+author|have\s+author|written\s+by|created\s+by|illustrated\s+by)\b',
            'by',
            message,
            flags=re.I
        )
        m = re.search(
            r'(.+?)\s+by\s+(.+)',
            message,
            flags=re.I
        )

        possible_authors = []

        if m:
            # print(m.group(1))
            possible_title = [f'{ent[1]}' for ent in self.nlu.predict_ner(m.group(1)) if ent[0] == 'MANGA_ENT']
            tmp_string = 'by ' + m.group(2)
            # print(tmp_string)
            possible_authors = [f'{ent[1]}' for ent in self.nlu.predict_ner(tmp_string) if ent[0] == 'MANGA_ENT']

        else:
            possible_title = [f'{ent[1]}' for ent in self.nlu.predict_ner(message) if ent[0] == 'MANGA_ENT']

        print('possible_title',possible_title)
        # print(m)
        print('possible_authors',possible_authors)

        if possible_title:
            query_filter = query_filter + f'''\'title:"{possible_title[0]}"'''
            query_filter = query_filter + f''' OR title_english:"{possible_title[0]}"'''

        if possible_authors:
            if possible_title:
                query_filter = query_filter + f''' AND authors:"{possible_authors[0]}"'''
            else:
                query_filter = query_filter + f'''\'authors:"{possible_authors[0]}"'''

        if query_filter == '':
            return query_filter

        query_filter = query_filter + "\'"
        if sort_rating:
            query_filter = query_filter + 'ORDER BY score DESC'
        return query_filter

    @staticmethod
    def _metadata_query_search(metadata_target: dict, search_col: list[str], sort_rating: bool) -> str:
        query_filter = f"""\'"""
        first_key = True
        tmp = ""
        for k, v in metadata_target.items():
            if k not in search_col:
                continue
            v = v.replace('"', '\\"')
            if first_key:
                tmp = f"{k}:\"{v}\""
                first_key = False
            else:
                tmp = tmp + f" OR {k}:\"{v}\""

        query_filter = query_filter + tmp + "\'"
        if sort_rating:
            query_filter = query_filter + "ORDER BY score DESC"

        return query_filter

    @staticmethod
    def _get_chat_history(chat_history:list[dict]|None, past_limit: int = 5):

        if not chat_history:
            context = f"""
                    <CONVERSATION_HISTORY>
                    No history data.
                    </CONVERSATION_HISTORY>"""

            return context

        recent_history = chat_history[-past_limit:]
        history_lines = []
        for item in recent_history:
            role = item.get('role', 'user')
            content = item.get('content', '')
            if content:
                history_lines.append(f'{role}: {content}')

        if not history_lines:
            context = f"""
                                <CONVERSATION_HISTORY>
                                No history data.
                                </CONVERSATION_HISTORY>"""

            return context

        temp_text = '\n'.join(history_lines)
        context = f"""
        <CONVERSATION_HISTORY>
        {temp_text}
        </CONVERSATION_HISTORY>"""

        return context

    def response_handler(self, message: str, stream: bool=False, verbose: bool=False, chat_history: list[dict]|None=None):
        self._verify_rag_available()
        context = ''
        sys_prompt = '''
             You are RAG chatbot about manga. You will be given context about manga and user question. Give answer based on context.
        '''.strip()
        intent = self.nlu.predict_intent(message)
        ner = self.nlu.predict_ner(message)
        print('intent', intent)
        print('ner', ner)

        if intent in ['metadata_lookup']:
            if verbose: print('RH-1')
            sys_prompt = '''
                         You are MangaBot, a friendly and knowledgeable manga assistant.

                        Your task is:
                        - Answer user question based on TARGET_MANGA section.
                        - If there is no data, do not add your own knowledge.


                         '''.strip()
            print('sql side')
            query_filter = self._metadata_lookup_query(message)
            print('query_filter1', query_filter)
            if query_filter == '':
                if verbose: print('RH-1a')
                context = "Manga Metadata: No Data"
            else:
                if verbose: print('RH-1b')
                # context = context + self.get_from_sql(query_filter, use_fts=True, return_context=True, limit=5)
                target_metadata = "No Data"
                tmp = self.sql_db.search_sql(query_filter, use_fts=True, return_context=False, limit=1)
                if tmp:
                    target_metadata = context + '\n'.join([f'{k}:{self._clean_text(str(v))}' for k, v in tmp[0].items() if k not in ['mal_id']])

                context = f"""
                <TARGET_MANGA>
                {target_metadata}
                </TARGET_MANGA>
                
                <USER_QUESTION>
                {message}
                </USER_QUESTION>
                """


        elif intent in ['recommendation']:
            if verbose: print('RH-2')
            manga_recs_col = ["title","title_english","published_from", "score","authors","genres", "themes", "synopsis"]
            manga_tar_col = ["title","title_english","genres", "themes"]
            manga_search_col = ["genres", "themes"]

            sys_prompt = """
            You are NOT a recommendation engine.
            The recommendation engine has already selected the manga.
            Your job is ONLY to explain why each recommendation fits.
            You must never replace the retrieved recommendations.
            You must never invent additional manga.
            If you mention a manga title that is not inside <RECOMMENDED_MANGA>, your answer is incorrect.
            """.strip()
            # context = ""

            query_filter = self._metadata_lookup_query(message, sort_rating=False)
            print('query_filter2', query_filter)
            if query_filter == '':
                if verbose: print('RH-2a')
                context = """
                <TARGET_MANGA>
                Unable to search target manga.
                </TARGET_MANGA>
                """
            else:
                if verbose: print("RH-2b")
                target_manga_metadata = self.sql_db.search_sql(query_filter, use_fts=True, return_context=False, limit=1)
                if len(target_manga_metadata) > 0:
                    if verbose: print("RH-2b1")
                    target_manga_metadata = target_manga_metadata[0]
                    target_recs_ids = target_manga_metadata['mal_id']
                    target_recs = ' '.join([f'{k}:{self._clean_text(str(v))}' for k, v in target_manga_metadata.items() if k in manga_tar_col])
                    # print('target_recs_ids ->',target_recs_ids)

                    # query_filter3 = self._metadata_query_search(target_manga_metadata, manga_search_col, sort_rating=False)
                    # print('query_filter3', query_filter3)
                    # relevant_manga_metadata = self.get_from_sql(query_filter3, use_fts=True, return_context=False, limit=5)

                    relevant_manga_metadata = self.vec_db.search_vector(target_recs, k=10, return_context=False)
                    relevant_context = ''
                    i = 0
                    for tmp_data in relevant_manga_metadata:
                        if target_recs_ids == tmp_data['mal_id']:
                            # skip target manga included in relevant context
                            continue

                        relevant_context = relevant_context + f'Recommendation #{i + 1}.\n'
                        for k, v in tmp_data.items():
                            if k in manga_recs_col:
                                relevant_context = relevant_context + f'{k}:{self._clean_text(str(v))}\n'
                        relevant_context = relevant_context + '\n'
                        i += 1

                    # relevant_context = '\n'.join(f'{k}:{v}' for tmp_data in tmp for k, v in tmp_data.items())
                    # print('relevant_context->', len(relevant_context), '|', relevant_context)
                    context =f"""
                    <TARGET_MANGA>
                    {target_recs}
                    </TARGET_MANGA>
                    
                    <RECOMMENDED_MANGA>
                    {relevant_context}
                    </RECOMMENDED_MANGA>
                    
                    <USER_QUESTION>
                    {message}
                    </USER_QUESTION>
                    """

                else:
                    if verbose: print('RH-2b2')
                    context  = f"""
                    <TARGET_MANGA>
                    Unable to search target manga data.
                    </TARGET_MANGA>
                    
                    <RECOMMENDED_MANGA>
                    Unable to give relevant manga data.
                    </RECOMMENDED_MANGA>
                    
                    <USER_QUESTION>
                    {message}
                    </USER_QUESTION>
                    """

        else:
            if verbose: print('RH-Default')

            sys_prompt = """
            You are MangaBot, a friendly and knowledgeable manga assistant.

            Your personality:
            - Be friendly and conversational.
            - Answer naturally, like chatting with another manga fan.
            - Give complete answers instead of one short sentence.
            - When appropriate, add a little extra helpful information.
            - Be enthusiastic about manga, but do not invent facts.
            - If the user asks casual questions (such as "Who are you?" or "How are you?"), respond naturally instead of redirecting to manga.
            - Keep responses around 2–5 sentences unless the user requests more detail.
            
            Your task is:
            - Response to user question/chat under USER_QUESTION section.
            - You may use CONVERSATION_HISTORY to read context if available. 
            """.strip()

            context = f"""
            <USER_QUESTION>
            {message}
            </USER_QUESTION>
            """
        # return context
        # context = self._clean_text(context)

        if chat_history:
            chat_hist = self._get_chat_history(chat_history)
            context = chat_hist + '\n' + context

        print('####### context ######')
        print(context)

        if stream:
            if verbose: print('RH-STREAM')
            def _response_generator():
                result = self.ollama_generate_chat(context, sys_prompt, stream=stream)
                for chunk in result:
                    yield chunk['message']['content']

            return _response_generator()
        else:
            if verbose: print('RH-FULLTEXT')
            gen_text = self.ollama_generate_chat(context, sys_prompt, stream=stream)['message']['content']

            return gen_text

    def response_handler_with_tool(self, messages: str, stream: bool=False, verbose: bool=False, chat_history: list[dict]|None=None, chat_history_limit: int = 5):
        self._verify_rag_available()
        context = ''
        sys_prompt = """
        You are MangaBot, a friendly and knowledgeable manga assistant.
        
        You can have normal conversations with the user without using any tools.
        
        Examples of messages that should NOT use any tool:
        - Hi
        - Hello
        - Who are you?
        - How are you?
        - Thank you
        - What can you do?
        - Tell me a joke
        
        For these messages, reply naturally as a friendly chatbot.
        
        --------------------------------------------------
        
        Use tools ONLY when the user is requesting manga information that requires searching the local database.
        
        Examples:
        - Find comedy manga.
        - Recommend manga similar to Berserk.
        - Show manga by Naoki Urasawa.
        - What genres does Monster have?
        - Find manga released in 2025.
        - Find a manga about an unemployed man who gets another chance at life.
        - Find a manga like Yuru Yuri.
        
        Choose the correct tool:
        
        • search_by_filters
        Use when the user specifies structured information such as:
        - title
        - author
        - genre
        - theme
        - publication year
        
        • search_by_synopsis
        Use when the user describes a story, characters, events, or plot instead of structured metadata.
        
        • recommend_by_title
        Use this tool when the user asks for manga recommendations similar to a specific manga.
        
        Tool-calling rules:
        - Only use tools when the user's request requires searching the manga database.
        - It is completely acceptable to answer without calling any tool.
        - Never guess missing tool arguments.
        - Only pass information explicitly provided by the user.
        - Leave unspecified optional arguments null.
        - After receiving tool results, answer only using the returned data.
        - If the tool returns no results, politely inform the user that no matching manga were found.
        """
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
        # print(template_messages)
        # raise ValueError
        response = chat(
            model="lukaspetrik/gemma3-tools:4b",
            messages=template_messages,
            tools=[search_by_filters,
                   search_by_synopsis,
                   recommend_by_title
                   ],
        )
        print_log("First LLM stage")
        # print(response.message)

        if response.message.tool_calls is None:
            if verbose: print('RH-STREAM NON TOOLS')
            def _response_generator():
                for i in range(0, len(response.message.content), 20):
                    yield response.message.content[i:i + 20]
                    # time.sleep(0.2)
            # def _response_generator():
            #     content = response.message.content.split()
            #     for chunk in content:
            #         yield chunk + " "

            return _response_generator()

        else:
            template_messages.append(response.message)
            for tool in response.message.tool_calls:
                tool_name = tool.function.name
                tool_output = None
                args = tool.function.arguments
                print(f"-> Gemma 3 triggered {tool_name} call with arguments: {args}")

                if tool_name == "search_by_filters":
                    # Execute the local function
                    tool_output = search_by_filters(self.sql_db,
                                                      title=args.get("title"),
                                                      genres=args.get("genres"),
                                                      authors=args.get("authors"),
                                                      themes=args.get("themes"),
                                                      released_year=args.get("released_year"),
                                                  )

                elif tool_name == "search_by_synopsis":
                    # Execute the local function
                    tool_output = search_by_synopsis(self.vec_db, query=args.get("query"))


                elif tool_name == "recommend_by_title":
                    # Execute the local function
                    tool_output = recommend_by_title(self.sql_db, title=args.get("title"))

                if tool_output is not None:
                    # Feed the function's result back into the chat history
                    print("TOOL OUTPUT", tool_output)
                    template_messages.append({
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": tool_output
                    })

            print()
            print("RH-AFTER TOOL CALLS")
            print(template_messages)
            if stream:
                if verbose: print('RH-STREAM_TOOL')
                def _response_generator():
                    result = chat(
                        model="lukaspetrik/gemma3-tools:4b",
                        messages=template_messages,
                        stream=stream,
                    )
                    for chunk in result:
                        yield chunk['message']['content']

                return _response_generator()
            else:
                if verbose: print('RH-FULLTEXT_TOOL')
                gen_text = chat(
                    model="lukaspetrik/gemma3-tools:4b", # "lukaspetrik/gemma3-tools:4b" "gemma3:4b"
                    messages=template_messages,
                    stream=stream,
                )['message']['content']

                return gen_text
