from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import Settings


def _apply_prefix(texts: list[str], prefix: str) -> list[str]:
    normalized_prefix = prefix.strip()

    if not normalized_prefix:
        return texts

    return [f"{normalized_prefix} {text}" for text in texts]


@lru_cache
def get_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def _validate_dimensions(vectors: list[list[float]], expected_dim: int) -> None:
    if not vectors:
        return

    current_dim = len(vectors[0])

    if current_dim != expected_dim:
        raise ValueError(
            "A dimensao do embedding nao bate com EMBEDDING_DIM. "
            f"Esperado: {expected_dim}. Recebido: {current_dim}."
        )


def embed_passages(texts: list[str], settings: Settings) -> list[list[float]]:
    model = get_embedding_model(settings.embedding_model)
    prepared_texts = _apply_prefix(texts, settings.embedding_document_prefix)
    vectors = model.encode(
        prepared_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    _validate_dimensions(vectors, settings.embedding_dim)
    return vectors


def embed_query(text: str, settings: Settings) -> list[float]:
    model = get_embedding_model(settings.embedding_model)
    prepared_text = _apply_prefix([text], settings.embedding_query_prefix)[0]
    vector = model.encode(
        prepared_text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).tolist()

    if len(vector) != settings.embedding_dim:
        raise ValueError(
            "A dimensao do embedding da pergunta nao bate com EMBEDDING_DIM. "
            f"Esperado: {settings.embedding_dim}. Recebido: {len(vector)}."
        )

    return vector
