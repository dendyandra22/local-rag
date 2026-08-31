import json


def _get_query_search_column(column: str, values: str | list[str], operator: str):
    query_filter = ""
    if isinstance(values, str):
        values = values.title()
        query_filter += f'''{column}:"{values}"'''
    if isinstance(values, list):
        query_filter = f"{column}:\"" + f"\" {operator} {column}:\"".join([data.title() for data in values]) + "\""

    return query_filter

def sql_get_by_columns(sqldb,
                    args: dict,
                    searchable_columns: list[str],
                    operator: str = "and",
                    use_fts=True,
                    return_context=True,
                    limit: int = 5
                    ):
    """
    Finds data matching a combination of specific columns and value mentioned by users.

    CRITICAL: Only provide arguments that the user EXPLICITLY mentioned in their prompt.
    DO NOT guess, hallucinate, or assume these values if they aren't provided by the user.
    """

    # if all(param is None for key, param in args.items()):
    # # if all(param not in ["title", "genres", "authors", "themes"] for param in args.keys()):
    #     raise AttributeError("params value is empty! fill at least one of it")

    operator = "AND" if operator in ["AND", "and"] else "OR"
    query_filter = "\'"

    search_filter = {}
    for col in searchable_columns:
        value = args.get(col)
        if value is not None:
            search_filter[col] = value

    print("search_filter",search_filter)
    for column, value in search_filter.items():

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


def _get_query_filter(filters: list, limit: int):
    """
    Make a SQL query for filters search.
    """
    def is_number(text: str):
        try:
            float(text)
            return True
        except ValueError:
            return False

    query = f"WHERE "
    op_convert = {"=": "LIKE"}

    filter_query = []
    for fil in filters:
        val = fil["value"]
        op = fil["operator"]

        # handle operation for string value column
        if not is_number(val):
            op = op_convert.get(fil["operator"], "LIKE")
            val = f"""\'%{fil["value"]}%\'"""

        tmp = f'''{fil["column"]} {op} {val}'''
        filter_query.append(tmp)

    query = query + ' AND '.join(filter_query).strip()
    query += f" LIMIT {limit}"
    return query

def sql_get_by_filter(sqldb, filter_list, limit, searchable_columns, return_context=True):
    """
    Find data by matching filter in columns.
    """

    # validate searchable column
    tmp = [fil["column"] for fil in filter_list if fil["column"] in searchable_columns]
    if len(tmp) == 0:
        raise Exception(f"Search column invalid! Please search in this column: {searchable_columns}")


    query_filter = _get_query_filter(filter_list, limit)
    res = sqldb.search_sql(query_filter, limit=limit, use_fts=False, return_context=return_context)

    return res