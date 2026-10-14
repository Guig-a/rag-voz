import asyncio
import logging
import time
from collections.abc import AsyncIterable

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    FlushSentinel,
    JobProcess,
    ModelSettings,
    TurnHandlingOptions,
    inference,
    llm,
)

from app.config import get_settings
from app.embeddings import get_embedding_model
from app.graph import run_rag_graph

load_dotenv()

logger = logging.getLogger("voice-agent")

FAILURE_REPLY = "Tive um problema para consultar os documentos agora. Pode repetir a pergunta?"
THINKING_REPLY = "Um momento, vou consultar os documentos. "


def _last_user_text(chat_ctx: llm.ChatContext) -> str:
    for item in reversed(chat_ctx.items):
        if getattr(item, "type", None) == "message" and item.role == "user":
            return (item.text_content or "").strip()
    return ""


class RagGraphLLM(llm.LLM):
    """O AgentSession so agenda respostas se houver um llm.LLM; quem responde e o llm_node."""

    def chat(self, **kwargs) -> llm.LLMStream:
        raise NotImplementedError("As respostas vem do grafo RAG em RagVoiceAgent.llm_node.")


class RagVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="Voce e um assistente de voz que responde em portugues, de forma curta.",
        )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str | FlushSentinel]:
        question = _last_user_text(chat_ctx)
        if not question:
            return

        yield THINKING_REPLY
        yield FlushSentinel()

        settings = get_settings()
        started = time.perf_counter()

        try:
            result = await asyncio.to_thread(
                run_rag_graph,
                settings,
                question=question,
                voice=True,
            )
            answer = result["answer"]
        except Exception:
            logger.exception("Falha no grafo RAG")
            answer = FAILURE_REPLY
            result = {}

        logger.info(
            "rag pergunta=%r reescrita=%r attempts=%s suficiente=%s chunks=%s tempo_rag=%.2fs",
            question,
            result.get("rewritten_question"),
            result.get("attempts"),
            result.get("context_is_enough"),
            result.get("used_chunk_ids"),
            time.perf_counter() - started,
        )

        yield answer


def prewarm(proc: JobProcess) -> None:
    get_embedding_model(get_settings().embedding_model)


server = AgentServer(setup_fnc=prewarm)


@server.rtc_session(agent_name="rag-voz")
async def rag_voice_session(ctx: agents.JobContext) -> None:
    settings = get_settings()

    session = AgentSession(
        stt=inference.STT(
            model=settings.voice_stt_model,
            language=settings.voice_language,
        ),
        tts=inference.TTS(
            model=settings.voice_tts_model,
            voice=settings.voice_tts_voice,
            language=settings.voice_language,
        ),
        llm=RagGraphLLM(),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
    )

    await session.start(room=ctx.room, agent=RagVoiceAgent())
    await session.say("Olá! Pode me perguntar sobre os seus documentos.")


if __name__ == "__main__":
    agents.cli.run_app(server)
