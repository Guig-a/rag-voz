"""Experimento 1: similaridade de cosseno calculada na mao com torch, comparada ao pgvector.

Uso: python -m experiments.01_cosine_by_hand "sua pergunta"
"""

import sys

import torch

from app.config import get_settings
from app.embeddings import embed_query
from app.retrieval import semantic_search
from experiments.common import load_chunks

DEFAULT_QUESTION = "Qual e o maior planeta do Sistema Solar?"
TOP_K = 5


def cosine_similarity(query: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    dot = matrix @ query
    norms = matrix.norm(dim=1) * query.norm()
    return dot / norms


def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    settings = get_settings()

    chunks, matrix = load_chunks(settings)
    query = torch.tensor(embed_query(question, settings), dtype=torch.float32)

    print(f"pergunta: {question}")
    print(f"matriz de chunks: shape={tuple(matrix.shape)} dtype={matrix.dtype}")
    print(f"vetor da pergunta: shape={tuple(query.shape)} norma={query.norm():.4f}")
    print(f"normas dos chunks: min={matrix.norm(dim=1).min():.4f} max={matrix.norm(dim=1).max():.4f}")

    scores = cosine_similarity(query, matrix)
    dot_only = matrix @ query
    print(f"maior diferenca entre cosseno e produto escalar: {(scores - dot_only).abs().max():.2e}")

    top = torch.topk(scores, k=min(TOP_K, len(chunks)))
    torch_ranking = [(chunks[i]["id"], float(scores[i])) for i in top.indices.tolist()]

    db_ranking = [
        (row["id"], float(row["score"]))
        for row in semantic_search(settings, query=question, limit=TOP_K)
    ]

    print()
    print(f"{'pos':>3}  {'torch id':>8}  {'torch score':>11}  {'pgvector id':>11}  {'pgvector score':>14}  {'diff':>9}")
    for position, ((torch_id, torch_score), (db_id, db_score)) in enumerate(
        zip(torch_ranking, db_ranking), start=1
    ):
        print(
            f"{position:>3}  {torch_id:>8}  {torch_score:>11.6f}  {db_id:>11}  "
            f"{db_score:>14.6f}  {abs(torch_score - db_score):>9.2e}"
        )

    same_order = [item[0] for item in torch_ranking] == [item[0] for item in db_ranking]
    print()
    print(f"mesma ordem de ids: {same_order}")
    best = chunks[int(top.indices[0])]
    print(f"melhor chunk ({best['document']}): {best['content'][:160]}...")


if __name__ == "__main__":
    main()
