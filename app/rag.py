from app.config import Settings
from app.graph import DONT_KNOW_ANSWER, run_rag_graph

__all__ = ["DONT_KNOW_ANSWER", "answer_question"]


def answer_question(
    settings: Settings,
    *,
    question: str,
    limit: int | None = None,
) -> dict:
    return run_rag_graph(
        settings,
        question=question,
        limit=limit,
    )
