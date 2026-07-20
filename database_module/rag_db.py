# from vectordb import VectorDB
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_PATH = BASE_DIR / "data"

class RAGDatabase:
    def __init__(self, db_name):
        self.db_name = db_name
        self.db_path = None
        self.conn = None
        # self.created_at = None

    def _check_db_exist(self):
        if self.db_path is None:
            return False
        return Path(self.db_path).exists()

    def _check_conn(self):
        if self.conn is None:
            return False
        else:
            return True


# class RAGDB:
#     def __init__(self, db_name, rebuild_db):
#         self.db_name = db_name
#         if rebuild_db:
#             self.sqm, self.vectordb = self._build_ragdb()
#
#         else:
#             self.sqm, self.vectordb = self._load_db()
#
#
#
#     def _build_ragdb(self):
#         sql = SQLDB().create_db(df)
#
#         vecdb = VectorDB().create_db(df)
#
#         return sql, vecdb
#
#     def _load_db(self):
#         sql = SQLDB().connect_db(self.db_name)
#         vecdb = VectorDB().connect_db(self.db_name)
#
#         return sql, vecdb