import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.chain import DISCLAIMER, build_chain, format_sources
from backend.schemas import QueryRequest, QueryResponse, UploadResponse
from backend.sessions import get_session_retriever, store_session_index

load_dotenv()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore_dir = os.environ.get("VECTORSTORE_DIR", "vectorstore")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FAISS.load_local(
        vectorstore_dir, embeddings, allow_dangerous_deserialization=True
    )
    base_retriever = index.as_retriever(search_kwargs={"k": 4})

    state["embeddings"] = embeddings
    state["chain"] = build_chain(base_retriever, get_session_retriever)
    yield
    state.clear()


app = FastAPI(title="Medical Info Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = state["chain"].invoke(
        {"question": request.question, "session_id": request.session_id}
    )
    return QueryResponse(
        answer=result["answer"],
        sources=format_sources(result["docs"]),
        disclaimer=DISCLAIMER,
    )


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not docs:
        raise HTTPException(
            status_code=400, detail="Couldn't extract any text from that PDF."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata = {"source": "Uploaded PDF", "title": file.filename, "url": ""}

    session_index = FAISS.from_documents(chunks, state["embeddings"])
    session_id = str(uuid.uuid4())
    store_session_index(session_id, session_index)

    return UploadResponse(session_id=session_id, chunk_count=len(chunks))
