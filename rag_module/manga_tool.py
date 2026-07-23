
def _get_query_search_column(column: str, values: str | list[str], operator: str):
    query_filter = ""
    if isinstance(values, str):
        values = values.title()
        query_filter += f'''{column}:"{values}"'''
    if isinstance(values, list):
        query_filter = f"{column}:\"" + f"\" {operator} {column}:\"".join([data.title() for data in values]) + "\""

    return query_filter

def search_by_filters(sqldb,
                    title: str = None,
                    genres: list[str] = None,
                    authors: list[str] = None,
                    themes: list[str] = None,
                    released_year: str = None,
                    operator: str = "and",
                    use_fts=True,
                    return_context=True,
                    limit: int = 5
                    ):
    """
    Finds manga matching a combination of specific filters like title, genres, themes, authors, released_year.

    CRITICAL: Only provide arguments that the user EXPLICITLY mentioned in their prompt.
    DO NOT guess, hallucinate, or assume these values if they aren't provided by the user.
    """

    if all(param is None for param in [title, genres, authors, themes, released_year]):
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
        "released_year": released_year,
    }

    for column, value in filters.items():
        if (value is None or len(value) < 1) or column == "title":
            continue

        if column == "released_year":
            column = "published_from"

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

def search_by_title(sqldb,
                     title: str,
                     use_fts=True,
                     limit: int = 5
                     ):
    """
    Looks up a specific manga by its title to get its metadata.
    Use when the user mentions a specific manga name (e.g., 'tell me about Naruto').
    """

    title = title.title()
    query_filter = f'''title:"{title}" OR title_english:"{title}"'''
    query_filter = "\'"+query_filter+"\'"

    res = sqldb.search_sql(query_filter, limit=limit, use_fts=use_fts, return_context=True)

    return res

def search_by_authors(sqldb,
                     authors: str | list[str],
                     use_fts=True,
                     limit: int = 5
                     ):
    """
    Finds manga written or illustrated by a specific author or artist.
    Use when the user asks for works by a person (e.g., 'manga by Naoki Urasawa and Rukako').
    """
    if authors is None or len(authors) == 0:
        raise AttributeError("metadata argument is empty!")

    query_filter = "\'" + _get_query_search_column(column='authors', values=authors, operator="AND") + "\'"

    res = sqldb.search_sql(query_filter, limit=limit, use_fts=use_fts, return_context=True)

    return res


def search_by_themes(sqldb,
                     themes: str | list[str],
                     use_fts=True,
                     limit: int = 5
                     ):
    """
    Finds manga matching specific plot themes or tropes.
    Use when the user specifies themes (e.g., 'school life').
    """
    if themes is None or len(themes) == 0:
        raise AttributeError("metadata argument is empty!")

    query_filter = "\'" + _get_query_search_column(column='themes', values=themes, operator="AND") + "\'"

    res = sqldb.search_sql(query_filter, limit=limit, use_fts=use_fts, return_context=True)

    return res

def search_by_genres(sqldb,
                     genres: str | list[str],
                     use_fts=True,
                     limit: int = 5
                     ):
    """
   Finds manga matching one or more specific genres.
    Use when the user asks for genre recommendations (e.g., 'give me romance sci-fi manga').
    """

    if genres is None or len(genres) == 0:
        raise AttributeError("metadata argument is empty!")


    query_filter = "\'" + _get_query_search_column(column='genres', values=genres, operator="AND") + "\'"
    query_filter += " ORDER BY score DESC"

    res = sqldb.search_sql(query_filter, limit=limit, use_fts=use_fts, return_context=True)

    return res

def search_by_synopsis(vec_db, query: str, top_k: int = 5):
    """
    Searches the vector database for manga based on a plot, story description, or character concept.

    Use this tool when the user describes a storyline, plot trope, character setup, or when they are
    trying to find a manga whose title they forgot (e.g., 'a story about a guy who gets reincarnated as a slime').
    """
    result = vec_db.search_vector(query, k=top_k, return_context=True)
    return result

def recommend_by_title(sqldb, title: str):
    """
    Recommend manga similar to a given manga title.

    Use this tool ONLY when the user explicitly mentions a manga title
    and asks for recommendations, similar manga, manga with the same
    genres, themes, style, or overall vibe.

    Examples:
    - Recommend manga like Berserk.
    - I like Monster. What should I read next?
    - Similar to One Piece.
    - Any manga with the same vibe as Yuru Yuri?

    Do NOT use this tool for:
    - Looking up manga information (chapters, synopsis, genres, etc.).
    - Searching by genres, themes, authors, or publication year.
    - Searching by plot description without a specific title.
    """
    result = ""
    args = search_by_filters(sqldb, title=title, use_fts=True, return_context=False, limit=1)

    # find similar manga like using target manga metadata
    if len(args) > 0:
        args = args[0]
        genres = args.get("genres", "")
        genres = genres.split("|")
        themes = args.get("themes", "")
        themes = themes.split("|")
        result = search_by_filters(sqldb,
                                        genres=genres,
                                        # themes=themes,
                                        use_fts=True,
                                        return_context=True,
                                        limit=5
                                        )

    return result