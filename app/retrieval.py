from app.chunking import chunk_text
from app.config import Settings
from app.db import ensure_schema, insert_document_chunks, search_chunks
from app.embeddings import embed_passages, embed_query


def ingest_document(settings: Settings, *, name: str, content: str) -> tuple[int, int]:
    ensure_schema(settings)

    chunks = chunk_text(
        text=content,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise ValueError("Nao ha texto suficiente para gerar chunks.")

    embeddings = embed_passages(chunks, settings)
    return insert_document_chunks(
        settings,
        name=name,
        chunks=chunks,
        embeddings=embeddings,
    )


def semantic_search(
    settings: Settings,
    *,
    query: str,
    limit: int | None = None,
) -> list[dict]:
    ensure_schema(settings)

    query_embedding = embed_query(query, settings)
    final_limit = limit or settings.search_default_k
    return search_chunks(
        settings,
        query_embedding=query_embedding,
        limit=final_limit,
    )
