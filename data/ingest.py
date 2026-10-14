from pathlib import Path

from app.config import get_settings
from app.db import document_exists, ensure_schema
from app.retrieval import ingest_document

DATA_DIR = Path(__file__).parent


def main() -> None:
    settings = get_settings()
    ensure_schema(settings)

    for path in sorted(DATA_DIR.glob("*.md")):
        if document_exists(settings, name=path.name):
            print(f"pulado (ja existe): {path.name}")
            continue

        document_id, chunk_count = ingest_document(
            settings,
            name=path.name,
            content=path.read_text(encoding="utf-8"),
        )
        print(f"ingerido: {path.name} document_id={document_id} chunks={chunk_count}")


if __name__ == "__main__":
    main()
