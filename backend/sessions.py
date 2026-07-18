"""In-memory storage for PDF-upload-scoped retrieval.

Each uploaded PDF gets its own small FAISS index, keyed by a generated
session_id, kept only in process memory — nothing is written to disk or
merged into the base vectorstore/. No TTL/eviction: acceptable at portfolio
scale, and a process restart clearing all sessions is the intended behavior
(uploaded content is never meant to persist).
"""

from langchain_community.vectorstores import FAISS

_sessions: dict[str, FAISS] = {}


def store_session_index(session_id: str, index: FAISS) -> None:
    _sessions[session_id] = index


def get_session_retriever(session_id: str):
    index = _sessions.get(session_id)
    if index is None:
        return None
    return index.as_retriever(search_kwargs={"k": 4})
