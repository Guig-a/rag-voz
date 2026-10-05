import json
import time
from pathlib import Path

from app.config import get_settings
from app.rag import DONT_KNOW_ANSWER, answer_question

QUESTIONS_PATH = Path(__file__).with_name("questions.jsonl")
NO_ANSWER_SENTINEL = "__NO_ANSWER__"


def load_cases() -> list[dict]:
    cases: list[dict] = []

    for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if not stripped:
            continue

        cases.append(json.loads(stripped))

    return cases


def contains_expected_chunk(sources: list[dict], expected_hint: str) -> bool:
    normalized_hint = expected_hint.strip().lower()

    if not normalized_hint:
        return False

    return any(normalized_hint in source["content"].lower() for source in sources)


def answer_matches(answer: str, expected_answer: str) -> bool:
    return expected_answer.strip().lower() in answer.strip().lower()


def should_retry(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE"))


def run_case(settings, question: str) -> dict:
    last_error: Exception | None = None

    for attempt in range(1, settings.eval_retry_attempts + 1):
        try:
            return answer_question(
                settings,
                question=question,
                limit=settings.search_default_k,
            )
        except Exception as error:
            last_error = error

            if attempt >= settings.eval_retry_attempts or not should_retry(error):
                raise

            delay = settings.eval_retry_delay_seconds * attempt
            print(
                f"Erro transitorio no caso '{question}' "
                f"(tentativa {attempt}/{settings.eval_retry_attempts}). "
                f"Nova tentativa em {delay:.1f}s."
            )
            time.sleep(delay)

    raise RuntimeError(f"Falha ao executar caso: {question}") from last_error


def main() -> None:
    settings = get_settings()
    cases = load_cases()

    if not cases:
        print("Nenhum caso encontrado em eval/questions.jsonl")
        return

    top_k_hits = 0
    answer_hits = 0
    refusal_hits = 0
    answer_cases = 0
    refusal_cases = 0
    infrastructure_failures = 0

    for index, case in enumerate(cases, start=1):
        try:
            result = run_case(settings, case["question"])
        except Exception as error:
            infrastructure_failures += 1
            print(f"[{index}] erro de infraestrutura: {error}")
            print("---")
            continue

        expected_answer = case["expected_answer"]
        expected_chunk_hint = case.get("expected_chunk_hint", "")
        source_dicts = [
            {
                "content": source["content"],
            }
            for source in result["sources"]
        ]

        top_k_hit = contains_expected_chunk(source_dicts, expected_chunk_hint)

        if expected_answer == NO_ANSWER_SENTINEL:
            refusal_cases += 1
            refused = result["answer"].strip().lower() == DONT_KNOW_ANSWER
            if refused:
                refusal_hits += 1
            status = "OK" if refused else "FALHOU"
            print(f"[{index}] recusa esperada: {status}")
            if settings.eval_delay_seconds > 0:
                time.sleep(settings.eval_delay_seconds)
            continue

        answer_cases += 1
        if top_k_hit:
            top_k_hits += 1

        matched = answer_matches(result["answer"], expected_answer)
        if matched:
            answer_hits += 1

        print(f"[{index}] top-k={'OK' if top_k_hit else 'FALHOU'}")
        print(f"[{index}] resposta={'OK' if matched else 'FALHOU'}")
        print(f"[{index}] best_score={result['best_score']}")
        print(f"[{index}] answer={result['answer']}")
        print("---")

        if settings.eval_delay_seconds > 0:
            time.sleep(settings.eval_delay_seconds)

    print("Resumo:")
    print(f"- casos totais: {len(cases)}")
    print(f"- chunk esperado no top-k: {top_k_hits}/{answer_cases}")
    print(f"- resposta esperada bateu: {answer_hits}/{answer_cases}")
    print(f"- recusas corretas: {refusal_hits}/{refusal_cases}")
    print(f"- falhas de infraestrutura/provedor: {infrastructure_failures}")


if __name__ == "__main__":
    main()
