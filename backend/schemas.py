from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    title: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    disclaimer: str
