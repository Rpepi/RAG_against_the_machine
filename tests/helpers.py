from src.indexing.chunking.common import MAX_CHUNK_SIZE, PROJECT_ROOT

VLLM = PROJECT_ROOT / "data" / "raw" / "vllm-0.10.1"


def read_raw(path) -> str:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def assert_valid(chunks, text: str, max_chunk_size: int = MAX_CHUNK_SIZE) -> None:
    """The chunks tile `text` exactly, in order, and none is too big."""
    assert chunks, "no chunk"
    pos = 0
    for chunk in chunks:
        assert chunk.first_character_index == pos, f"gap or overlap at {pos}"
        assert chunk.text == text[chunk.first_character_index:chunk.last_character_index]
        assert chunk.size == len(chunk.text) > 0
        assert chunk.size <= max_chunk_size
        assert chunk.file_path.startswith("data/raw/")
        pos = chunk.last_character_index
    assert pos == len(text), "the end of the file is missing"


def cut_offsets(chunks) -> list[int]:
    """Offsets where one chunk ends and the next starts."""
    return [chunk.first_character_index for chunk in chunks[1:]]


def code_block(size: int) -> str:
    """A fenced code block of exactly `size` characters."""
    return "```\n" + "x" * (size - 9) + "\n```\n"
