from app.config import Settings
from app.llm import generate_grounded_answer
from app.retrieval import semantic_search

DONT_KNOW_ANSWER = "nao encontrei isso nos documentos"


def answer_question(
    settings: Settings,
    *,
    question: str,
    limit: int | None = None,
) -> dict:
    chunks = semantic_search(
        settings,
        query=question,
        limit=limit,
    )

    if not chunks:
        return {
            "answer": DONT_KNOW_ANSWER,
            "sources": [],
            "used_chunk_ids": [],
            "best_score": None,
        }

    best_score = float(chunks[0]["score"])

    if best_score < settings.rag_min_score:
        return {
            "answer": DONT_KNOW_ANSWER,
            "sources": [],
            "used_chunk_ids": [],
            "best_score": best_score,
        }

    answer = generate_grounded_answer(
        settings,
        question=question,
        chunks=chunks,
    )

    return {
        "answer": answer,
        "sources": chunks,
        "used_chunk_ids": [chunk["id"] for chunk in chunks],
        "best_score": best_score,
    }
