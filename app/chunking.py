def _find_chunk_end(text: str, start: int, max_end: int) -> int:
    if max_end >= len(text):
        return len(text)

    window = text[start:max_end]
    last_space = window.rfind(" ")

    if last_space <= 0:
        return max_end

    return start + last_space


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized_text = " ".join(text.split())

    if not normalized_text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap precisa ser menor que chunk_size.")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0

    while start < len(normalized_text):
        max_end = min(len(normalized_text), start + chunk_size)
        end = _find_chunk_end(normalized_text, start, max_end)
        chunk = normalized_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(normalized_text):
            break

        start += step

    return chunks
