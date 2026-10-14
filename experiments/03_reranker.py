"""Experimento 3: reordenar o top-k da busca vetorial com um cross-encoder e medir se melhora.

Mede so a recuperacao (sem LLM): em que posicao o chunk esperado de cada caso do
eval/questions.jsonl aparece antes e depois do reranker.

Uso: python -m experiments.03_reranker
"""

import time

from sentence_transformers import CrossEncoder

from app.config import get_settings
from app.retrieval import semantic_search
from experiments.common import find_rank, load_answerable_cases, load_refusal_cases

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CANDIDATES = 10


def rerank(model: CrossEncoder, question: str, chunks: list[dict]) -> list[dict]:
    scores = model.predict([(question, chunk["content"]) for chunk in chunks])
    ranked = sorted(zip(chunks, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [{**chunk, "rerank_score": float(score)} for chunk, score in ranked]


def summarize(ranks: list[int | None]) -> str:
    total = len(ranks)
    hit_at_1 = sum(1 for rank in ranks if rank == 1)
    hit_at_3 = sum(1 for rank in ranks if rank is not None and rank <= 3)
    mrr = sum(1 / rank for rank in ranks if rank is not None) / total
    return f"hit@1={hit_at_1}/{total} hit@3={hit_at_3}/{total} MRR={mrr:.3f}"


def main() -> None:
    settings = get_settings()
    model = CrossEncoder(RERANK_MODEL)

    vector_ranks: list[int | None] = []
    rerank_ranks: list[int | None] = []
    rerank_seconds = 0.0

    print(f"{'vetor':>5}  {'rerank':>6}  pergunta")
    for case in load_answerable_cases():
        candidates = semantic_search(settings, query=case["question"], limit=CANDIDATES)

        started = time.perf_counter()
        reranked = rerank(model, case["question"], candidates)
        rerank_seconds += time.perf_counter() - started

        vector_rank = find_rank(candidates, case["expected_chunk_hint"])
        rerank_rank = find_rank(reranked, case["expected_chunk_hint"])
        vector_ranks.append(vector_rank)
        rerank_ranks.append(rerank_rank)
        print(f"{vector_rank or '-':>5}  {rerank_rank or '-':>6}  {case['question']}")

    print()
    print(f"busca vetorial: {summarize(vector_ranks)}")
    print(f"com reranker:   {summarize(rerank_ranks)}")
    print(f"tempo medio do reranker: {rerank_seconds / len(rerank_ranks) * 1000:.0f} ms por pergunta")

    print()
    print("melhor score por pergunta (separacao entre 'tem resposta' e 'nao tem'):")
    print(f"{'cosseno':>8}  {'rerank':>7}  pergunta")
    for label, cases in (("com resposta", load_answerable_cases()), ("sem resposta", load_refusal_cases())):
        print(f"-- {label}")
        for case in cases:
            candidates = semantic_search(settings, query=case["question"], limit=CANDIDATES)
            reranked = rerank(model, case["question"], candidates)
            print(
                f"{float(candidates[0]['score']):>8.4f}  {reranked[0]['rerank_score']:>7.2f}  "
                f"{case['question']}"
            )


if __name__ == "__main__":
    main()
