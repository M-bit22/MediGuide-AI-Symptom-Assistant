import os
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache, SQLiteCache

def setup_cache(cache_type: str):
    """
    Configure global LangChain cache based on user selection.
    """
    if cache_type == "In-Memory (RAM)":
        set_llm_cache(InMemoryCache())
    elif cache_type == "SQLite (Disk)":
        db_path = os.path.join(os.getcwd(), "langchain_cache.db")
        set_llm_cache(SQLiteCache(database_path=db_path))
    else:
        set_llm_cache(None)