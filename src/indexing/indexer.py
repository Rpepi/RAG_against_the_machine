from pathlib import Path
from src.indexing.chunking.models import (
    Chunck,
)
from src.indexing.chunking.python_chunking import python_chunker
from src.indexing.chunking.markdown_chunking import markdown_chunker
from src.indexing.chunking.text_chunking import text_chunker
from src.indexing.bm25 import BM25Index
from tqdm import tqdm



def collect_chunks(raw_dir: Path, max_chunk_size: int) -> list[Chunck]:
    chunks: list[Chunck] = []
    raw_dir = Path(raw_dir)
    for path in tqdm(sorted(raw_dir.rglob("*")), desc="Chunking"):
        if not path.is_file():
            continue
        if path.suffix == ".py":
            chunks.extend(python_chunker(path, max_chunk_size))
        if path.suffix== ".md":
            chunks.extend(markdown_chunker(path, max_chunk_size))
        if path.suffix == ".txt":
            chunks.extend(text_chunker(path, max_chunk_size))
    return chunks


def indexer(raw_dir: Path = "data/raw", max_chunk_size: int = 2000, index_path: str = "data/processed/bm25_index.json") -> None:
    chunks = collect_chunks(raw_dir, max_chunk_size)
    index = BM25Index()
    index.build(chunks)
    index.save(index_path)



if __name__ == "__main__":
    indexer()