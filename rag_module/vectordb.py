import faiss
# from sentence_transformers import SentenceTransformer
import pandas as pd
import re
# from tqdm import tqdm
import pickle
import ollama
import numpy as np
from pathlib import Path

from util.logger import print_log

class VectorDB:
    def __init__(self, db_name: str,
                 inference_type: str = 'ollama',
                 source_path: str = None,
                 rebuild_db: bool = False,
                 max_rows: int = 10000,
                 col_filter: list = None
                 ):

        self.db_name = db_name
        self.inference_type = inference_type
        self.vector_db = None
        self.vector_index = None
        # self.source_path = source_path
        # self.rebuild_db = rebuild_db
        # self.max_rows = max_rows

        if rebuild_db:
            file_path = Path(source_path)
            # Check if the file specifically exists
            if not file_path.is_file():
                print_log(f"source file does not exist!")
                return

            self._create_db(source_path, max_rows, col_filter=col_filter)

        else:
            self.vector_db, self.vector_index = self._load_db()

    @staticmethod
    def _make_document(dataframe):
        pattern = r'\s+'
        for data in dataframe.itertuples():
            yield f'''
                title: {data.title}
                title_english: {data.title_english}
                score: {data.score}
                authors: {data.authors}
                genres: {data.genres}
                themes: {data.themes}
                synopsis: {re.sub(pattern, " ", data.synopsis)}
                '''.strip()

    def _embed_docs(self, docs: list):
        vectors = None
        if self.inference_type == 'ollama':
            embeddings = ollama.embed(
                model='nomic-embed-text',
                input=docs
            )
            vectors = np.array(
                embeddings["embeddings"],
                dtype=np.float32
            )

        return vectors

    def _create_db(self, file_path: str, max_rows: int, col_filter: list):
        print_log(f"preparing new vector DB with name {self.db_name}.faiss")

        df = pd.read_csv(file_path)
        df = df.loc[:max_rows, col_filter]
        df.dropna(subset=['title'], inplace=True)
        fillcol = {
            'title_english': '-',
            'type': '-',
            'chapters': 0,
            'volumes': 0,
            'status': '-',
            'published_from': '-',
            'score': 0,
            'authors': '-',
            'genres': '-',
            'themes': '-',
            'synopsis': '-'}
        df.fillna(fillcol, inplace=True)
        docs = list(self._make_document(df))

        print_log(f'start embedding {len(docs)} documents')
        vectors = self._embed_docs(docs)
        print("embed shape", vectors.shape)

        print_log(f'store embedding data as data/{self.db_name}.faiss')
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(
            index,
            f"data/{self.db_name}.faiss"
        )
        print_log(f'store embedding index data as data/{self.db_name}_mapping.pkl')
        id_mapping = df["mal_id"].tolist()
        with open(f"data/{self.db_name}_mapping.pkl", "wb") as f:
            pickle.dump(id_mapping, f)

    def _load_db(self):
        vdb = faiss.read_index(f'data/{self.db_name}.faiss')
        with open(f"data/{self.db_name}_mapping.pkl", "rb") as f:
            vdb_idx = pickle.load(f)

        print_log(f'successfully load {self.db_name} vector database')
        return vdb, vdb_idx

    def search_vector(self, query):
        mal_ids = []

        if self.inference_type == 'ollama':
            query_emb = ollama.embed(
                model='nomic-embed-text',
                input=query
            )

            query_vector = np.array(
                query_emb['embeddings'],
                dtype=np.float32
            )

            distances, indices = self.vector_db.search(
                query_vector,
                k=5
            )
            mal_ids = [
                self.vector_index[i]
                for i in indices[0]
            ]

        return mal_ids