# Medical Info Assistant

RAG-grounded health-information Q&A chatbot. Answers general questions
("what is condition X", "common symptoms of Y") from cited public health
sources (MedlinePlus, CDC, WHO), plus optional PDF upload for session-scoped
Q&A over an additional document. **Not diagnostic and not personalized
medical advice** — always cites sources, always shows a disclaimer, and
refuses to diagnose or recommend treatment for a described personal
situation.

## Stack

- Python, LangChain (`langchain-openai`, `langchain-community`)
- LLM: `gpt-4o-mini` · Embeddings: `text-embedding-3-small`
- Vector store: FAISS, persisted to disk, built by `ingestion/build_index.py`
- Backend: FastAPI (`/query`, `/upload`, `/health`)
- Frontend: Streamlit, calls the backend over HTTP
- Dependency management: `uv`

## Local development

```
uv sync
cp .env.example .env   # fill in OPENAI_API_KEY
uv run python -m ingestion.build_index   # builds vectorstore/
uv run uvicorn backend.main:app --reload
uv run streamlit run frontend/app.py
```
