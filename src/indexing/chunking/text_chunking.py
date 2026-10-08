from pathlib import Path
from src.indexing.chunking.models import Chunck
from src.indexing.chunking.common import (
    DEFAULT_MAX_CHUNK_SIZE,
    make_chunk,
    read_source,
    refine,
)


def text_chunker(path: Path, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[Chunck]:
    """A text file has no structure: one section, cut by paragraph then by size."""
    text, file_path = read_source(path)
    if not text.strip():
        return []
    return refine([make_chunk(file_path, 0, text, Path(path).name)], max_chunk_size=max_chunk_size)


if __name__ == "__main__":
    from src.indexing.chunking.common import PROJECT_ROOT

    for chunk in text_chunker(PROJECT_ROOT / "data" / "raw" / "vllm-0.10.1" / "requirements" / "cuda.txt"):
        print(f"[{chunk.first_character_index}:{chunk.last_character_index}] {chunk.title!r}")
        print(chunk.text)
