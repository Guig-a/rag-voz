import logging

import psycopg
from psycopg.rows import dict_row

from app.config import Settings

logger = logging.getLogger(__name__)


def _format_vector(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def ensure_schema(settings: Settings) -> None:
    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        CREATE TABLE IF NOT EXISTS documents (
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS chunks (
          id SERIAL PRIMARY KEY,
          document_id INT REFERENCES documents(id) ON DELETE CASCADE,
          position INT NOT NULL,
          content TEXT NOT NULL,
          embedding vector({settings.embedding_dim}) NOT NULL
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
        ON chunks
        USING hnsw (embedding vector_cosine_ops);
        """,
    ]

    with psycopg.connect(settings.database_url) as connection:
        with connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement)

        connection.commit()


def insert_document_chunks(
    settings: Settings,
    *,
    name: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> tuple[int, int]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks e embeddings precisam ter o mesmo tamanho.")

    with psycopg.connect(settings.database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (name)
                VALUES (%s)
                RETURNING id;
                """,
                (name,),
            )
            document_id = cursor.fetchone()[0]

            for position, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                cursor.execute(
                    """
                    INSERT INTO chunks (document_id, position, content, embedding)
                    VALUES (%s, %s, %s, %s::vector);
                    """,
                    (document_id, position, chunk, _format_vector(embedding)),
                )

        connection.commit()

    logger.info(
        "Documento ingerido com %s chunks. document_id=%s",
        len(chunks),
        document_id,
    )
    return document_id, len(chunks)


def search_chunks(
    settings: Settings,
    *,
    query_embedding: list[float],
    limit: int,
) -> list[dict]:
    with psycopg.connect(settings.database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                  id,
                  document_id,
                  position,
                  content,
                  1 - (embedding <=> %s::vector) AS score
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (
                    _format_vector(query_embedding),
                    _format_vector(query_embedding),
                    limit,
                ),
            )
            rows = cursor.fetchall()

    logger.info(
        "Busca retornou chunks=%s",
        [(row["id"], round(row["score"], 4)) for row in rows],
    )
    return rows
