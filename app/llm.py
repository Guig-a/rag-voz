import logging
import time

from anthropic import Anthropic
from google import genai
from google.genai import types

from app.config import Settings

logger = logging.getLogger(__name__)

GROUNDING_SYSTEM_PROMPT = """
Voce e um assistente de RAG.
Responda somente com base no contexto fornecido.
Se o contexto nao bastar para responder com seguranca, responda exatamente:
"nao encontrei isso nos documentos"
Nao invente fatos e nao use conhecimento externo.
""".strip()

TRANSIENT_LLM_MARKERS = ("429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


def _build_context(chunks: list[dict]) -> str:
    blocks: list[str] = []

    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    f"[chunk_id={chunk['id']} document_id={chunk['document_id']} position={chunk['position']} score={float(chunk['score']):.4f}]",
                    chunk["content"],
                ]
            )
        )

    return "\n\n---\n\n".join(blocks)


def is_transient_llm_error(error: Exception) -> bool:
    message = str(error).upper()
    return any(marker in message for marker in TRANSIENT_LLM_MARKERS)


def _complete_once(
    settings: Settings,
    *,
    system: str,
    user: str,
    token_limit: int,
) -> str:
    provider = settings.llm_provider.strip().lower()

    if provider == "anthropic":
        client = Anthropic(api_key=settings.llm_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=token_limit,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ]
        answer = "\n".join(parts).strip()
    elif provider == "gemini":
        client = genai.Client(api_key=settings.llm_api_key)
        response = client.models.generate_content(
            model=settings.llm_model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=token_limit,
            ),
        )
        answer = (response.text or "").strip()
    else:
        raise ValueError("LLM_PROVIDER invalido. Use 'gemini' ou 'anthropic'.")

    if not answer:
        raise ValueError("O LLM nao retornou uma resposta em texto.")

    return answer


def complete(
    settings: Settings,
    *,
    system: str,
    user: str,
    max_tokens: int | None = None,
) -> str:
    if not settings.llm_api_key or not settings.llm_model or not settings.llm_provider:
        raise ValueError(
            "Configure LLM_PROVIDER, LLM_API_KEY e LLM_MODEL no .env para usar /ask."
        )

    token_limit = max_tokens or settings.llm_max_tokens
    last_error: Exception | None = None

    for attempt in range(1, settings.llm_retry_attempts + 1):
        try:
            return _complete_once(
                settings,
                system=system,
                user=user,
                token_limit=token_limit,
            )
        except ValueError:
            raise
        except Exception as error:
            last_error = error

            if attempt >= settings.llm_retry_attempts or not is_transient_llm_error(error):
                raise

            delay = settings.llm_retry_delay_seconds * attempt
            logger.warning(
                "LLM transitorio (tentativa %s/%s). Nova tentativa em %.1fs. erro=%s",
                attempt,
                settings.llm_retry_attempts,
                delay,
                error,
            )
            time.sleep(delay)

    raise RuntimeError("Falha ao chamar o LLM.") from last_error


def generate_grounded_answer(
    settings: Settings,
    *,
    question: str,
    chunks: list[dict],
) -> str:
    context = _build_context(chunks)
    user_prompt = "\n".join(
        [
            "<contexto>",
            context,
            "</contexto>",
            "",
            "<pergunta>",
            question,
            "</pergunta>",
            "",
            "Responda em portugues e cite apenas o que o contexto sustenta.",
        ]
    )

    return complete(
        settings,
        system=GROUNDING_SYSTEM_PROMPT,
        user=user_prompt,
    )
