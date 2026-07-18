import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.chain import DISCLAIMER, build_chain, format_sources
from backend.schemas import QueryRequest, QueryResponse

load_dotenv()

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    vectorstore_dir = os.environ.get("VECTORSTORE_DIR", "vectorstore")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    index = FAISS.load_local(
        vectorstore_dir, embeddings, allow_dangerous_deserialization=True
    )
    retriever = index.as_retriever(search_kwargs={"k": 4})
    state["chain"] = build_chain(retriever)
    yield
    state.clear()


app = FastAPI(title="Medical Info Assistant", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    result = state["chain"].invoke(request.question)
    return QueryResponse(
        answer=result["answer"],
        sources=format_sources(result["docs"]),
        disclaimer=DISCLAIMER,
    )
