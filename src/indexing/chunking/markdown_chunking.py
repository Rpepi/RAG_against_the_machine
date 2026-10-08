import re
from pathlib import Path
from src.indexing.chunking.models import Chunck
from src.indexing.chunking.common import (
    DEFAULT_MAX_CHUNK_SIZE,
    PROJECT_ROOT,
    Block,
    read_source,
    refine,
    split_at,
)


TITLE_RE = re.compile(r"#{1,3} +(.+?)\s*$")
FENCE = "```"


def code_blocks(text: str) -> list[Block]:
    """(start, end) of every fenced code block of the file, fences included."""
    blocks: list[Block] = []
    start: int | None = None
    pos = 0
    for line in text.split("\n"):
        end = min(pos + len(line) + 1, len(text))
        if line.lstrip().startswith(FENCE):
            if start is None:
                start = pos
            else:
                blocks.append((start, end))
                start = None
        pos += len(line) + 1
    if start is not None:
        blocks.append((start, len(text)))
    return blocks


def split_title(text: str, file_path: str, blocks: list[Block]) -> list[Chunck]:
    """One chunk per section, a section starting at a `#`, `##` or `###` line."""
    titles: list[tuple[int, str]] = []
    pos = 0
    for line in text.split("\n"):
        match = TITLE_RE.match(line)
        if match and not any(start <= pos < end for start, end in blocks):
            titles.append((pos, match.group(1)))
        pos += len(line) + 1
    return split_at(text, file_path, titles, "")


def markdown_chunker(path: Path, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[Chunck]:
    text, file_path = read_source(path)
    if not text.strip():
        return []
    blocks = code_blocks(text)
    return refine(split_title(text, file_path, blocks), blocks, max_chunk_size)


if __name__ == "__main__":
    chunks = markdown_chunker(Path(PROJECT_ROOT) / "data" / "raw" / "vllm-0.10.1" / "docs" / "design" / "plugin_system.md")
    for n, chunk in enumerate(chunks, 1):
        print(f"chunk {n} [{chunk.first_character_index}:{chunk.last_character_index}] {chunk.title!r}")
        print(chunk.text + "\n")
