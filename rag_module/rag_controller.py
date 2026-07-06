import re

from rag_module.response import Response

class RAGController:
    def __init__(self,
                 db_name: str,
                 timestamp: str,
                 inference_type: str = None,
                 retrain_model: bool = False,
                 ):
        inference_type = inference_type if inference_type else 'ollama'
        self.response = Response(db_name, timestamp, inference_type, retrain_model)
        self.nlu_model = self.response.nlu_model

    def _metadata_lookup_query(self, message, sort_rating=True):
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
            possible_title = [f'{ent[1]}' for ent in self.nlu_model.predict_ner(m.group(1)) if ent[0] == 'MANGA_ENT']
            tmp_string = 'by ' + m.group(2)
            # print(tmp_string)
            possible_authors = [f'{ent[1]}' for ent in self.nlu_model.predict_ner(tmp_string) if ent[0] == 'MANGA_ENT']

        else:
            possible_title = [f'{ent[1]}' for ent in self.nlu_model.predict_ner(message) if ent[0] == 'MANGA_ENT']

        print('possible_title',possible_title)
        # print(m)
        print('possible_authors',possible_authors)

        if possible_title:
            query_filter = query_filter + f'''\'(title:"{possible_title[0]}"'''
            query_filter = query_filter + f''' OR title_english:"{possible_title[0]}")'''

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


    def response_handler(self, message, stream=False, verbose=False, chat_history=None):
        context = ''
        sys_prompt = '''
             You are RAG chatbot about manga. You will be given context about manga and user question. Give answer based on context.
        '''.strip()
        intent = self.response.nlu_model.predict_intent(message)
        ner = self.response.nlu_model.predict_ner(message)
        print('intent', intent)
        print('ner', ner)

        if intent in ['metadata_lookup']:
            if verbose: print('RH-1')
            sys_prompt = '''
                         You are RAG chatbot about manga. You will be given context about manga and user question. Give answer based on context. Do not give outside the context given.

                         '''.strip()
            print('sql side')
            context = '''Context:\n\n'''
            query_filter = self._metadata_lookup_query(message)
            print('query_filter1', query_filter)
            if query_filter == '':
                if verbose: print('RH-1a')
                context = context + 'No context provided'
            else:
                if verbose: print('RH-1b')
                context = context + self.response.get_from_sql(query_filter, use_fts=True, return_context=True, limit=5)


        elif intent in ['general','recommendation']:
            if verbose: print('RH-2')
            # sys_prompt = '''
            #             You are a manga recommendation assistant.
            #
            #             You retrieve Target Recommend contain manga that user like, and Relevant Context contain similar manga.
            #             Recommend the manga in Relevant Context by explain metadata and reason why they may be relevant for user.
            #             If no Target Recommend data, politely explain to user and ask to give more specific or detail about manga they like.
            #              '''.strip()
            sys_prompt = """
            You are a manga recommendation assistant.

            You will receive two sections:

            1. Target Manga
            - The manga that the user likes or is asking about.
            - This section may be empty.

            2. Relevant Context
            - A list of manga retrieved from the database because they are semantically similar to the Target Manga or the user's request.

            Instructions:
            - Recommend only manga from the Relevant Context.
            - Briefly describe each recommendation using its available metadata (such as genres, themes, score, synopsis, or author).
            - Explain why each recommendation may appeal to the user based on similarities to the Target Manga or the user's request.
            - Do not invent manga or information that is not present in the context.
            - If Relevant Context is empty, politely state that no suitable recommendations were found.
            - If Target Manga is empty, ask the user to mention a manga they enjoy or describe the type of manga they are looking for.
            - Format the answer as a friendly recommendation list.
            """.strip()
            # context = '''Relevant context:\n\n'''

            query_filter = self._metadata_lookup_query(message, sort_rating=False)
            print('query_filter2', query_filter)
            if query_filter == '':
                if verbose: print('RH-2a')
                context = 'Target Recommend:\n\nUnable to search target manga.'
            else:
                if verbose: print('RH-2b')
                tmp = self.response.get_from_sql(query_filter, use_fts=True, return_context=False, limit=1)
                if len(tmp) > 0:
                    if verbose: print('RH-2b1')
                    target_recs_ids = tmp[0]['mal_id']
                    target_recs = '\n'.join([f'{k}:{v}' for k, v in tmp[0].items()])
                    print('target_recs_ids ->',target_recs_ids)

                    tmp = self.response.get_from_vectordb(target_recs, k=5, return_context=False)
                    # drop first idx to remove target manga included in relevant context
                    # tmp = tmp[1:]

                    relevant_context = ''
                    for i,tmp_data in enumerate(tmp):
                        if target_recs_ids == tmp_data['mal_id']:
                            continue

                        relevant_context = relevant_context + f'{i + 1}.\n'
                        for k, v in tmp_data.items():
                            relevant_context = relevant_context + f'{k}:{v}\n'
                        relevant_context = relevant_context + '\n'

                    # relevant_context = '\n'.join(f'{k}:{v}' for tmp_data in tmp for k, v in tmp_data.items())
                    print('relevant_context->', len(relevant_context), '|', relevant_context)
                    context = context + f'Target Recommend:\n\n{target_recs}\n\n Relevant Context:\n\n{relevant_context}'

                else:
                    if verbose: print('RH-2b2')
                    context = 'Target Recommend:\n\nTarget manga data not available.'

        if stream:
            if verbose: print('RH-STREAM')
            def _response_generator():
                result = self.response.generate_chat(message, context, sys_prompt, stream=stream, chat_history=chat_history)
                for chunk in result:
                    yield chunk['message']['content']

            return _response_generator()
        else:
            if verbose: print('RH-FULLTEXT')
            result = self.response.generate_chat(message, context, sys_prompt, stream=stream, chat_history=chat_history)['message']['content']

            return result

