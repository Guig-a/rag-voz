import logging
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.llm import complete, generate_grounded_answer
from app.retrieval import semantic_search

logger = logging.getLogger(__name__)

DONT_KNOW_ANSWER = "nao encontrei isso nos documentos"

REWRITE_SYSTEM_PROMPT = """
Voce reescreve perguntas para melhorar busca semantica em documentos.
Responda apenas com a pergunta reescrita, sem aspas e sem explicacao.
Mantenha o idioma original.
Nao invente conceitos novos. Preserve os termos-chave da pergunta original.
Fique curto: uma frase.
""".strip()

GRADE_SYSTEM_PROMPT = """
Voce avalia se o contexto recuperado basta para responder a pergunta.
Responda exatamente SIM ou NAO.
SIM somente se o contexto contem a informacao necessaria.
NAO se o contexto e irrelevante, incompleto ou so parece relacionado.
""".strip()


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    chunks: list[dict]
    context_is_enough: bool
    attempts: int
    answer: str
    search_limit: int


def _rewrite_user_prompt(state: RAGState) -> str:
    if state["attempts"] <= 1:
        angle = "Reescreva de forma mais concreta e curta, mantendo os termos-chave."
    else:
        angle = "Tente outro angulo com sinonimos, mas sem perder os termos originais."

    return "\n".join(
        [
            f"Pergunta original: {state['question']}",
            f"Tentativa: {state['attempts']}",
            angle,
        ]
    )


def _grade_user_prompt(question: str, chunks: list[dict]) -> str:
    preview = "\n\n".join(
        f"- score={float(chunk['score']):.4f}: {chunk['content']}" for chunk in chunks
    )
    return "\n".join(
        [
            f"Pergunta: {question}",
            "",
            "Contexto recuperado:",
            preview or "(nenhum chunk)",
        ]
    )


def _parse_grade(raw: str) -> bool:
    normalized = raw.strip().upper()
    first_token = normalized.split()[0].strip(".,:;!") if normalized else "NAO"

    if first_token in {"SIM", "YES"}:
        return True
    if first_token in {"NAO", "NÃO", "NO"}:
        return False

    return False


def build_rag_graph(settings: Settings, *, voice: bool = False):
    def rewrite_question(state: RAGState) -> dict:
        attempts = int(state.get("attempts", 0)) + 1
        original = state["question"]

        if voice and attempts == 1:
            logger.info("rewrite_question attempts=1 voz: usando a pergunta original")
            return {"attempts": attempts, "rewritten_question": original}

        try:
            rewritten = complete(
                settings,
                system=REWRITE_SYSTEM_PROMPT,
                user=_rewrite_user_prompt({**state, "attempts": attempts}),
                max_tokens=120,
            )
            rewritten = rewritten.strip().strip('"') or original
        except Exception as error:
            logger.warning(
                "rewrite_question falhou; usando a pergunta original. erro=%s",
                error,
            )
            rewritten = original

        logger.info("rewrite_question attempts=%s rewritten=%s", attempts, rewritten)
        return {"attempts": attempts, "rewritten_question": rewritten}

    def retrieve(state: RAGState) -> dict:
        query = state["rewritten_question"] or state["question"]
        chunks = semantic_search(
            settings,
            query=query,
            limit=state.get("search_limit"),
        )
        scores = [round(float(chunk["score"]), 4) for chunk in chunks]
        logger.info("retrieve query=%s scores=%s", query, scores)
        return {"chunks": chunks}

    def grade_context(state: RAGState) -> dict:
        chunks = state.get("chunks") or []
        if not chunks:
            logger.info("grade_context suficiente=False motivo=sem_chunks")
            return {"context_is_enough": False}

        best_score = float(chunks[0]["score"])
        if best_score < settings.rag_min_score:
            logger.info(
                "grade_context suficiente=False motivo=score_baixo best_score=%s",
                best_score,
            )
            return {"context_is_enough": False}

        if voice:
            logger.info("grade_context suficiente=True motivo=voz_score best_score=%s", best_score)
            return {"context_is_enough": True}

        try:
            raw_grade = complete(
                settings,
                system=GRADE_SYSTEM_PROMPT,
                user=_grade_user_prompt(state["question"], chunks),
                max_tokens=16,
            )
        except Exception as error:
            logger.warning(
                "grade_context falhou; aceitando pelo score. best_score=%s erro=%s",
                best_score,
                error,
            )
            return {"context_is_enough": True}

        enough = _parse_grade(raw_grade)
        logger.info(
            "grade_context suficiente=%s raw=%s best_score=%s",
            enough,
            raw_grade,
            best_score,
        )
        return {"context_is_enough": enough}

    def generate(state: RAGState) -> dict:
        answer = generate_grounded_answer(
            settings,
            question=state["question"],
            chunks=state["chunks"],
            voice=voice,
        )
        logger.info("generate ok")
        return {"answer": answer}

    def say_dont_know(state: RAGState) -> dict:
        logger.info("say_dont_know after attempts=%s", state.get("attempts"))
        return {"answer": DONT_KNOW_ANSWER}

    def route_after_grade(
        state: RAGState,
    ) -> Literal["generate", "rewrite_question", "say_dont_know"]:
        if state.get("context_is_enough"):
            return "generate"
        if int(state.get("attempts", 0)) < settings.graph_max_attempts:
            return "rewrite_question"
        return "say_dont_know"

    graph = StateGraph(RAGState)
    graph.add_node("rewrite_question", rewrite_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_context", grade_context)
    graph.add_node("generate", generate)
    graph.add_node("say_dont_know", say_dont_know)

    graph.add_edge(START, "rewrite_question")
    graph.add_edge("rewrite_question", "retrieve")
    graph.add_edge("retrieve", "grade_context")
    graph.add_conditional_edges(
        "grade_context",
        route_after_grade,
        {
            "generate": "generate",
            "rewrite_question": "rewrite_question",
            "say_dont_know": "say_dont_know",
        },
    )
    graph.add_edge("generate", END)
    graph.add_edge("say_dont_know", END)

    return graph.compile()


def run_rag_graph(
    settings: Settings,
    *,
    question: str,
    limit: int | None = None,
    voice: bool = False,
) -> dict:
    compiled = build_rag_graph(settings, voice=voice)
    final_state = compiled.invoke(
        {
            "question": question,
            "rewritten_question": "",
            "chunks": [],
            "context_is_enough": False,
            "attempts": 0,
            "answer": "",
            "search_limit": limit or settings.search_default_k,
        }
    )

    chunks = final_state.get("chunks") or []
    context_is_enough = bool(final_state.get("context_is_enough"))
    sources = chunks if context_is_enough else []
    best_score = float(chunks[0]["score"]) if chunks else None

    return {
        "answer": final_state.get("answer") or DONT_KNOW_ANSWER,
        "sources": sources,
        "used_chunk_ids": [chunk["id"] for chunk in sources],
        "best_score": best_score,
        "rewritten_question": final_state.get("rewritten_question") or question,
        "attempts": int(final_state.get("attempts") or 0),
        "context_is_enough": context_is_enough,
    }
