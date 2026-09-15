import sqlite3
from pathlib import Path
import pandas as pd
# from datetime import datetime

from util.logger import print_log
from database_module.rag_db import RAGDatabase

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data"

class SQLDB(RAGDatabase):
    def __init__(self, db_name):
        super().__init__(db_name)

    def connect_db(self):
        self.db_path = DATABASE_PATH / f"{self.db_name}.db"
        if self._check_db_exist():
            # self.conn = sqlite3.connect(self.db_path)
            print_log(f"{self.db_name}.db connection is available.")

        else:
            self.db_path = None
            raise FileNotFoundError(f"{self.db_name}.db does not exist.")

    def create_db(self, df: pd.DataFrame):
        if self._check_db_exist():
            print_log(f"rebuilding SQL DB {self.db_name}.db")
        else:
            print_log(f"creating SQL DB {self.db_name}.db")
            self.db_path = DATABASE_PATH / f"{self.db_name}.db"

        print_log(f"total rows SQLDB: {df.shape[0]}")
        Path("data").mkdir(exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        df.to_sql(self.db_name, conn, if_exists="replace", index=False)

        # creating full text search (FTS) SQL DB
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {self.db_name}_fts;")

        # create a virtual FTS5 table
        column_query = ''
        unindexed_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
        for col in df.columns:
            if col in unindexed_columns:
                column_query += f'{col} UNINDEXED, '
            else:
                column_query += f'{col}, '
        column_query = column_query.strip()
        if column_query[-1] == ',':
            column_query = column_query[:-1]
        cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.db_name}_fts USING fts5({column_query});")

        # copy existing data into the new FTS table
        column_query = ', '.join(df.columns.tolist())
        cursor.execute(f"INSERT INTO {self.db_name}_fts({column_query}) SELECT {column_query} FROM {self.db_name};")
        conn.commit()

        print_log(f"{self.db_name}.db created successfully.")


    def search_sql(self, query_filter: str, use_fts: bool, return_context: bool, column_selection: str = None, limit: int = None, verbose: bool = False):

        column_selection = column_selection if column_selection else "*"

        if use_fts:
            if query_filter == '':
                raise ValueError("query_filter cannot be empty if use_fts is True!")
            query = f'''
            SELECT {column_selection} FROM {self.db_name}_fts
            WHERE {self.db_name}_fts MATCH ''' + query_filter

        else:
            query = f'''SELECT {column_selection} FROM {self.db_name} ''' + query_filter

        if verbose: print_log(f"SQL query: {query}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query)
                if limit is None:
                    all_rows = cursor.fetchall()
                else:
                    all_rows = cursor.fetchmany(limit)

                col_list = [desc[0] for desc in cursor.description]
            finally:
                cursor.close()

        if return_context:
            context = ''
            for row in all_rows:
                context = context + ' '.join(f'{col}: {val}\n' for col, val in zip(col_list, row))
                context += "\n\n"

            return context

        data = [{k: v for k, v in zip(col_list, row)} for row in all_rows]
        return data
