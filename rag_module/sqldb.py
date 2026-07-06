import sqlite3
import pandas as pd
from pathlib import Path

from util.logger import print_log

class SQLDB:
    def __init__(self,db_name: str, source_path: str = None, rebuild_db: bool = False, max_rows: int = 10000):
        self.db_name = db_name
        self.db_path = f"data/{db_name}.db"
        self.conn = None
        # self.source_path = source_path
        # self.rebuild_db = rebuild_db
        # self.max_rows = max_rows

        if rebuild_db:
            file_path = Path(source_path)
            # Check if the file specifically exists
            if not file_path.is_file():
                print_log(f"source file does not exist!")
                return

            self.conn = self._create_db(source_path, max_rows)

        else:
            self.sql_connect().close()


    def sql_connect(self):
        db_path = Path(self.db_path)
        if db_path.is_file():
            conn = sqlite3.connect(db_path)
            print_log(f"{self.db_name}.db connected successfully.")
            return conn
        else:
            print_log(f"{self.db_name}.db does not exist.")
            raise FileNotFoundError(f"{self.db_name}.db does not exist.")

    def _create_db(self, source_path: str, max_rows: int = 10000):
        print_log(f"creating SQL DB {self.db_name}.db")
        col_list = ['mal_id', 'title', 'title_english', 'type', 'chapters', 'volumes', 'status', 'published_from',
                    'published_to', 'score', 'authors', 'genres', 'themes', 'synopsis']

        df = pd.read_csv(source_path)
        df = df.loc[:max_rows, col_list]
        df.dropna(subset=['title'], inplace=True)

        conn = sqlite3.connect(f"data/{self.db_name}.db")
        df.to_sql(self.db_name, conn, if_exists="replace", index=False)

        # creating full text search (FTS) SQL DB
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {self.db_name}_fts;")

        # create a virtual FTS5 table
        cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS {self.db_name}_fts USING fts5(\
                       mal_id UNINDEXED, title, title_english, type, chapters UNINDEXED, \
                       volumes UNINDEXED, status, published_from, published_to, score UNINDEXED, \
                       authors, genres, themes, synopsis\
                       );")

        # copy existing data into the new FTS table
        cursor.execute(f"INSERT INTO {self.db_name}_fts(mal_id, title, title_english, type, chapters, volumes, status,\
         published_from, published_to, score, authors, genres, themes, synopsis) SELECT mal_id, title, \
         title_english, type, chapters, volumes, status, published_from, published_to, score, authors, \
         genres, themes, synopsis \
         FROM {self.db_name};")
        conn.commit()

        return conn


    def search_manga(self, query, param, limit):
        conn = self.sql_connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, param)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchmany(limit)
        finally:
            conn.close()

        result = [
            dict(zip(columns, row))
            for row in rows
        ]

        return result
