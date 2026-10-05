from anthropic import Anthropic
from google import genai
from google.genai import types

from app.config import Settings

GROUNDING_SYSTEM_PROMPT = """
Voce e um assistente de RAG.
Responda somente com base no contexto fornecido.
Se o contexto nao bastar para responder com seguranca, responda exatamente:
"nao encontrei isso nos documentos"
Nao invente fatos e nao use conhecimento externo.
""".strip()


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


def generate_grounded_answer(
    settings: Settings,
    *,
    question: str,
    chunks: list[dict],
) -> str:
    if not settings.llm_api_key or not settings.llm_model or not settings.llm_provider:
        raise ValueError(
            "Configure LLM_PROVIDER, LLM_API_KEY e LLM_MODEL no .env para usar /ask."
        )

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

    provider = settings.llm_provider.strip().lower()

    if provider == "anthropic":
        client = Anthropic(api_key=settings.llm_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=GROUNDING_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
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
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=GROUNDING_SYSTEM_PROMPT,
                max_output_tokens=settings.llm_max_tokens,
            ),
        )
        answer = (response.text or "").strip()
    else:
        raise ValueError(
            "LLM_PROVIDER invalido. Use 'gemini' ou 'anthropic'."
        )

    if not answer:
        raise ValueError("O LLM nao retornou uma resposta em texto.")

    return answer
