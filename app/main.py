from fastapi import Depends, FastAPI, HTTPException

from app.config import Settings, get_settings
from app.retrieval import ingest_document, semantic_search
from app.schemas import (
    IngestRequest,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(
        payload: IngestRequest,
        app_settings: Settings = Depends(get_settings),
    ) -> IngestResponse:
        try:
            document_id, chunks_created = ingest_document(
                app_settings,
                name=payload.name,
                content=payload.content,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return IngestResponse(
            document_id=document_id,
            chunks_created=chunks_created,
        )

    @app.post("/search", response_model=SearchResponse)
    def search(
        payload: SearchRequest,
        app_settings: Settings = Depends(get_settings),
    ) -> SearchResponse:
        try:
            matches = semantic_search(
                app_settings,
                query=payload.query,
                limit=payload.k,
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        return SearchResponse(
            query=payload.query,
            results=[
                SearchResult(
                    chunk_id=match["id"],
                    document_id=match["document_id"],
                    position=match["position"],
                    content=match["content"],
                    score=float(match["score"]),
                )
                for match in matches
            ],
        )

    return app


app = create_app()
