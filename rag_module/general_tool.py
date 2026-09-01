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


def _get_query_filter(filters: list):
    """
    Make a SQL query for filters search.
    """
    def is_number(text: str):
        try:
            float(text)
            return True
        except ValueError:
            return False

    if len(filters) == 0: return ""
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
    # query += f" LIMIT {limit}"
    return query

def sql_get_by_filter(sqldb, filter_list, limit, searchable_columns, return_context=True):
    """
    Find data by matching filter in columns.
    """

    # validate searchable column
    tmp = [fil["column"] for fil in filter_list if fil["column"] in searchable_columns]
    if len(tmp) == 0:
        raise Exception(f"Search column invalid! Please search in this column: {searchable_columns}")

    limit = limit if limit is not None else 5

    query_filter = _get_query_filter(filter_list)
    res = sqldb.search_sql(query_filter, limit=limit, use_fts=False, return_context=return_context)

    return res

def sql_get_basic_aggregation(sqldb, aggregation_list: list[dict], filter_list: list[dict], searchable_columns: list, limit: int, return_context: bool = True):
    """
    Do basic SQL aggregation like SUM, AVG, COUNT, MIN, MAX.
    """
    # # validate searchable column
    # tmp = [fil["column"] for fil in filter_list if fil["column"] in searchable_columns]
    # if len(tmp) == 0:
    #     raise ValueError(f"Search column invalid! Please search in this column: {searchable_columns}")

    limit = limit if limit is not None else 5

    # create AGG query
    tmp_query = []
    for agg_data in aggregation_list:
        agg_func = agg_data["aggregation_function"]
        target_col = agg_data["target_col"]
        query = f"{agg_func}({target_col})"
        if agg_func == "COUNT_DISTINCT":
            query = f"COUNT(DISTINCT {target_col})"
        tmp_query.append(query)

    agg_query = ", ".join(tmp_query)

    # create condition/filter query
    query_filter = _get_query_filter(filter_list)

    # use column_selection to search_sql since agg is different from basic SQL query
    res = sqldb.search_sql(query_filter,
                           column_selection=agg_query,
                           limit=limit,
                           use_fts=False,
                           return_context=return_context)

    return res


# ============================= tool calling schema =====================

def _generate_column_property(columns: list):
    properties = {}
    for col in columns:
        properties[col] = {
            "type": "string",
            "description": f"The exact value for {col} mentioned by the user. Leave null if not mentioned."
        }

    return properties


def generate_toolcall_schema(columns: list):
    column_property = _generate_column_property(columns)

    schema = [
        {
            "type": "function",
            "function": {
                "name": "search_by_columns",
                "description": "Finds data matching specific columns. ONLY provide arguments explicitly mentioned by the user.",
                "parameters": {
                    "type": "object",
                    "properties": column_property,
                    "required": []
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_by_filter",
                "description": "Filters the dataset based on conditions (like greater than, less than, or equals) and returns a specific number of rows.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filters": {
                            "type": "array",
                            "description": "A list of conditions to apply to the data.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {
                                        "type": "string",
                                        "description": "The database column to filter.",
                                        "enum": columns
                                    },
                                    "operator": {
                                        "type": "string",
                                        "description": "The comparison operator.",
                                        "enum": ["=", ">", "<", ">=", "<="]
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "The value to compare against. (e.g., 'games', '7.5')"
                                    }
                                },
                                "required": ["column", "operator", "value"]
                            }
                        },
                        "limit": {
                            "type": "integer",
                            "description": "The maximum number of rows to return. Default to 10 if the user doesn't specify.",
                            "default": 10
                        }
                    },
                    "required": ["filters"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "basic_aggregation",
                "description": "Calculates math like MIN, MAX, AVG, SUM, or COUNT on a specific numeric column.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "aggregation_list": {
                            "type": "array",
                            "description": "A list of aggregation to apply to the data.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "aggregation_function": {
                                        "type": "string",
                                        "description": "The type of math to do.",
                                        "enum": ["MIN", "MAX", "AVG", "SUM", "COUNT"]
                                    },
                                    "target_col": {
                                        "type": "string",
                                        "description": "The comparison operator.",
                                        "enum": columns
                                    }
                                },
                                "required": ["aggregation_function", "target_col"]
                            }
                        },
                        "filters": {
                            "type": "array",
                            "description": "A list of conditions to apply to the data.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {
                                        "type": "string",
                                        "description": "The database column to filter.",
                                        "enum": columns
                                    },
                                    "operator": {
                                        "type": "string",
                                        "description": "The comparison operator.",
                                        "enum": ["=", ">", "<", ">=", "<="]
                                    },
                                    "value": {
                                        "type": "string",
                                        "description": "The value to compare against. (e.g., 'games', '7.5')"
                                    }
                                },
                                "required": ["column", "operator", "value"]
                            }
                        },
                        "limit": {
                            "type": "integer",
                            "description": "The maximum number of rows to return. Default to 10 if the user doesn't specify.",
                            "default": 10
                        }
                    },
                    "required": ["aggregation_list"]
                }
            }
        }
    ]

    return schema
