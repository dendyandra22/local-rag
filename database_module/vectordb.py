import faiss
# from sentence_transformers import SentenceTransformer
import re
import pickle
import ollama
import numpy as np
from pathlib import Path

import pandas as pd

from util.logger import print_log
from database_module.rag_db import RAGDatabase
from database_module.sqldb import SQLDB

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data"

class VectorDB(RAGDatabase):
    def __init__(self, db_name, inference_type: str = None):
        super().__init__(db_name)
        self.inference_type = inference_type if inference_type else "ollama"
        self.conn_index = None

    def connect_db(self):
        self.db_path = DATABASE_PATH / f"{self.db_name}.faiss"
        if self._check_db_exist():
            vdb = faiss.read_index(f'data/{self.db_name}.faiss')
            with open(f"data/{self.db_name}_mapping.pkl", "rb") as f:
                vdb_idx = pickle.load(f)

            print_log(f'successfully load {self.db_name} vector database')
            self.conn = vdb
            self.conn_index = vdb_idx
        else:
            self.db_path = None
            raise AttributeError(f'{self.db_name} vector database not found')

    @staticmethod
    def _make_document(dataframe):
        pattern = r'\s+'
        columns = list(dataframe.columns)
        columns.remove('mal_id')
        for data in dataframe.itertuples():
            yield '\n\n'.join([f"{col}:\n{getattr(data, col)}" for col in columns])

    # @staticmethod
    # def _clean_text(text):
    #     text = text.replace('|', '\n')
    #     text = re.sub(r"\(Source:.*?\)", " ", text)
    #     text = re.sub(r"\s+", " ", text).strip()
    #
    #     return text

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

    def create_db(self, df: pd.DataFrame):
        if self._check_db_exist():
            print_log(f"rebuilding Vector DB {self.db_name}.faiss")

        else:
            print_log(f"creating Vector DB {self.db_name}.faiss")

        self.db_path = DATABASE_PATH / f"{self.db_name}.faiss"
        docs = list(self._make_document(df))
        assert len(docs) == df.shape[0]
        # print(docs[0])
        # raise ValueError
        print_log(f'start embedding {len(docs)} documents')
        vectors = self._embed_docs(docs)
        print("vector embed shape", vectors.shape)

        Path("data").mkdir(exist_ok=True)
        print_log(f'save embedding data as data/{self.db_name}.faiss')
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        faiss.write_index(
            index,
            f"data/{self.db_name}.faiss"
        )
        print_log(f'save embedding index data as data/{self.db_name}_mapping.pkl')
        id_mapping = df["mal_id"].tolist()
        with open(f"data/{self.db_name}_mapping.pkl", "wb") as f:
            pickle.dump(id_mapping, f)

        self.conn = index
        self.conn_index = id_mapping
        print_log(f"{self.db_name}.faiss created successfully.")

    @staticmethod
    def _clean_text(text):
        text = text.replace('|', ', ')
        text = re.sub(r"\(Source:.*?\)", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def search_vector(self, query: str, k: int, return_context: bool = False):
        if not self._check_conn():
            raise AttributeError(f'{self.db_name} vector database connection not initiated.')
        mal_ids = []
        distances = []

        if self.inference_type == 'ollama':
            query = self._clean_text(query)

            query_emb = ollama.embed(
                model='nomic-embed-text',
                input=query
            )

            query_vector = np.array(
                query_emb['embeddings'],
                dtype=np.float32
            )

            distances, indices = self.conn.search(
                query_vector,
                k=k
            )
            mal_ids = [
                self.conn_index[i]
                for i in indices[0]
            ]
            distances = distances[0]

            mal_ids = [str(ids) for ids in mal_ids[:k]]
            # print('response mal_id', mal_ids)
            # print('distances mal_id', distances)

            # make conn to sql db
            sql_db = SQLDB(self.db_name)
            sql_db.connect_db()
            context = ''
            if len(mal_ids) > 0:
                placeholders = ",".join(mal_ids)
                query_filter = f'WHERE mal_id IN ({placeholders})'
                context = sql_db.search_sql(query_filter, use_fts=False, limit=k, return_context=return_context)

            return context
