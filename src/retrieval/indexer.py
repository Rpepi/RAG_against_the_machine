from collections.abc import Callable
from pathlib import Path

from tqdm import tqdm

from src.chunking.markdown_chunking import markdown_chunker
from src.chunking.python_chunking import python_chunker
from src.chunking.text_chunking import text_chunker
from src.config import DEFAULT_MAX_CHUNK_SIZE, PROJECT_ROOT
from src.data_models.models import Chunck
from src.retrieval.bm25 import DEFAULT_INDEX_PATH, BM25Index
from src.utils.errors import RagError

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"

DASHES = ("-", "\u2013", "\u2014")  # hyphen, en dash, em dash

Chunker = Callable[[Path, int], list[Chunck]]
CHUNKERS: dict[str, Chunker] = {
    ".py": python_chunker,
    ".md": markdown_chunker,
    ".txt": text_chunker,
}


def collect_chunks(raw_dir: Path | str, max_chunk_size: int) -> list[Chunck]:
    """Chunk every indexable file (.py, .md, .txt) under `raw_dir`.

    Raises:
        RagError: `raw_dir` is not a folder, or `max_chunk_size` is invalid.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        hint = ""
        if raw_dir.name.startswith(DASHES):
            hint = " (it looks like an option written with a wrong dash)"
        raise RagError(f"Corpus folder not found: {raw_dir}{hint}")
    if isinstance(max_chunk_size, bool) or not isinstance(max_chunk_size, int):
        raise RagError(
            f"max_chunk_size must be an integer, got {max_chunk_size!r}"
        )
    if max_chunk_size < 1:
        raise RagError(f"max_chunk_size must be at least 1: {max_chunk_size}")
    chunks: list[Chunck] = []
    files = [path for path in sorted(raw_dir.rglob("*")) if path.is_file()]
    for path in tqdm(files, desc="Chunking"):
        chunker = CHUNKERS.get(path.suffix)
        if chunker is not None:
            chunks.extend(chunker(path, max_chunk_size))
    return chunks


def indexer(
    raw_dir: Path | str = DEFAULT_RAW_DIR,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> None:
    """Chunk the corpus, build the BM25 index and save it.

    Raises:
        RagError: the corpus folder is missing or holds nothing to index.
    """
    chunks = collect_chunks(raw_dir, max_chunk_size)
    if not chunks:
        raise RagError(f"Nothing to index: no .py, .md or .txt in {raw_dir}")
    index = BM25Index()
    index.build(chunks)
    try:
        index.save(Path(index_path))
    except OSError as e:
        raise RagError(f"Cannot write {index_path}: {e.strerror}") from e
    print(f"Ingestion complete! Indices saved under {index_path}")
