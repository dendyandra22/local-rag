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