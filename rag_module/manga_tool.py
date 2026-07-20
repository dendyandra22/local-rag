import re
from pathlib import Path
import pandas as pd

from nlu_module.nlu_component import NLUComponent
# from rag_module.response import Response
from util.logger import print_log
from database_module.vectordb import VectorDB
from database_module.sqldb import SQLDB


def _get_query_search_column(column: str, values: str | list[str], operator: str):
    query_filter = ""
    if isinstance(values, str):
        values = values.title()
        query_filter += f'''{column}:"{values}"'''
    if isinstance(values, list):
        query_filter = f"{column}:\"" + f"\" {operator} {column}:\"".join([data.title() for data in values]) + "\""

    return query_filter


# def search_metadata(sqldb,
#                     # args: dict,
#                     operator: str = "or",
#                     use_fts=True,
#                     return_context=False,
#                     limit: int = 5
#                     ):

def search_metadata(sqldb,
                    title: str = None,
                    genres: str | list[str] = None,
                    authors: str | list[str] = None,
                    themes: str | list[str] = None,
                    operator: str = "or",
                    use_fts=True,
                    return_context=False,
                    limit: int = 5
                    ):
    """
    Searches the local SQLite database for manga metadata.

    CRITICAL: Only provide arguments that the user EXPLICITLY mentioned in their prompt.
    DO NOT guess, hallucinate, or assume these values if they aren't provided by the user.
    """

    if all(param is None for param in [title, genres, authors, themes]):
    # if all(param not in ["title", "genres", "authors", "themes"] for param in args.keys()):
        raise AttributeError("title,genres,authors,themes is empty! fill at least one of it")

    operator = "AND" if operator in ["AND", "and"] else "OR"
    query_filter = "\'"

    if title:
        title = title.title()
        query_filter += f'''title:"{title}" OR title_english:"{title}" {operator} '''

    filters = {
        "title": title,
        "genres": genres,
        "authors": authors,
        "themes": themes,
    }

    for column, value in filters.items():
        if value is None:
            continue

        if query_filter == "\'":
            query_filter = query_filter + _get_query_search_column(column, value, operator="AND") + f" {operator} "
        else:
            query_filter = query_filter + _get_query_search_column(column, value, operator="AND") + f" {operator} "

    if query_filter.split()[-1] in ["AND", "OR"]:
        query_filter = " ".join(query_filter.split()[:-1]).strip()

    query_filter += "\'"
    query_filter = query_filter.strip()

    if query_filter == "\'\'":
        raise ValueError('no query for search in SQL')

    res = sqldb.search_sql(query_filter, limit=limit, use_fts=use_fts, return_context=return_context)

    return res

def search_by_genres(sqldb,
                     genres: str | list[str],
                     use_fts=True,
                     return_context=False,
                     limit: int = 5
                     ):
    """
    Find similar manga from local SQLite database given by metadata input. This function will return metadata of similar manga ordered by rating descending.
    """

    if genres is None or len(genres) == 0:
        raise AttributeError("metadata argument is empty!")


    query_filter = "\'" + _get_query_search_column(column='genres', values=genres, operator="AND") + "\'"
    query_filter += " ORDER BY score DESC"

    res = sqldb.search_sql(query_filter, limit=limit, use_fts=use_fts, return_context=return_context)

    return res



