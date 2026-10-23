import pytest
from helpers import VLLM, assert_valid, read_raw
from src.chunking.markdown_chunking import markdown_chunker
from src.chunking.python_chunking import python_chunker
from src.chunking.text_chunking import text_chunker

CHUNKERS = {
    ".py": python_chunker,
    ".md": markdown_chunker,
    ".txt": text_chunker,
}


def corpus_files():
    return sorted(
        path
        for path in VLLM.rglob("*")
        if path.suffix in CHUNKERS and path.is_file()
    )


@pytest.mark.skipif(not VLLM.exists(), reason="vllm corpus not found")
@pytest.mark.parametrize("max_size", [500, 2000])
def test_no_chunk_of_the_corpus_exceeds_max_chunk_size(max_size):
    """One chunk over the limit invalidates the whole moulinette output."""
    for path in corpus_files():
        text = read_raw(path)
        if not text.strip():
            continue
        chunks = CHUNKERS[path.suffix](path, max_size)
        assert_valid(chunks, text, max_size)
