from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    session_id: str | None = None


class SourceRef(BaseModel):
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    disclaimer: str


class UploadResponse(BaseModel):
    session_id: str
    chunk_count: int
