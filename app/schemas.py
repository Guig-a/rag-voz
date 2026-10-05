from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    name: str = Field(min_length=1)
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    document_id: int
    chunks_created: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    chunk_id: int
    document_id: int
    position: int
    content: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
