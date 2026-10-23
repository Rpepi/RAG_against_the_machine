from pathlib import Path
from src.data_models.models import Chunck
from src.config import DEFAULT_MAX_CHUNK_SIZE
from src.chunking.common import (
    make_chunk,
    read_source,
    refine,
)


def text_chunker(
    path: Path | str, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE
) -> list[Chunck]:
    """A text file has no structure: one section, then `refine`."""
    text, file_path = read_source(path)
    if not text.strip():
        return []
    return refine(
        [make_chunk(file_path, 0, text, Path(path).name)],
        max_chunk_size=max_chunk_size,
    )


if __name__ == "__main__":
    from src.config import PROJECT_ROOT

    for chunk in text_chunker(
        PROJECT_ROOT
        / "data"
        / "raw"
        / "vllm-0.10.1"
        / "requirements"
        / "cuda.txt"
    ):
        print(
            f"[{chunk.first_character_index}:"
            f"{chunk.last_character_index}] {chunk.title!r}"
        )
        print(chunk.text)
