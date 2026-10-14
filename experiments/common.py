import json
from pathlib import Path

import psycopg
import torch
from psycopg.rows import dict_row

from app.config import Settings
from eval.run_eval import NO_ANSWER_SENTINEL, load_cases, normalize

OUTPUT_DIR = Path(__file__).parent / "output"


def load_chunks(settings: Settings) -> tuple[list[dict], torch.Tensor]:
    """Le todos os chunks com o embedding salvo e devolve (metadados, matriz n x dim)."""
    with psycopg.connect(settings.database_url, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id, c.content, d.name AS document, c.embedding::text AS embedding
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.id;
                """
            )
            rows = cursor.fetchall()

    matrix = torch.tensor(
        [json.loads(row.pop("embedding")) for row in rows],
        dtype=torch.float32,
    )
    return rows, matrix


def load_answerable_cases() -> list[dict]:
    return [case for case in load_cases() if case["expected_answer"] != NO_ANSWER_SENTINEL]


def load_refusal_cases() -> list[dict]:
    return [case for case in load_cases() if case["expected_answer"] == NO_ANSWER_SENTINEL]


def find_rank(chunks: list[dict], expected_hint: str) -> int | None:
    """Posicao (1-based) do primeiro chunk que contem o trecho esperado."""
    hint = normalize(expected_hint)
    for position, chunk in enumerate(chunks, start=1):
        if hint and hint in normalize(chunk["content"]):
            return position
    return None
