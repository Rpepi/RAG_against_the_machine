import math
import re
from collections.abc import Sequence
from pathlib import Path
from src.data_models.models import Chunck
from src.config import (
    DEFAULT_MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    PROJECT_ROOT,
    RAW_PREFIX,
)


BLANK_LINE_RE = re.compile(r"\n(?:[ \t\r]*\n)+")

Block = tuple[int, int]


def read_source(path: Path | str) -> tuple[str, str]:
    """Text of the file, and the `file_path` to give to its chunks.

    The `file_path` is the path from the project root, under
    `data/raw/`. A file
    already inside `data/raw/` keeps its path, so the prefix is never doubled.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    relative = path.resolve().relative_to(PROJECT_ROOT)
    if relative.parts[:2] != tuple(RAW_PREFIX.split("/")):
        relative = Path(RAW_PREFIX) / relative
    return text, relative.as_posix()


def check_max_chunk_size(max_chunk_size: int) -> None:
    """Raise ValueError if `max_chunk_size` cannot hold any chunk."""
    if max_chunk_size < 1:
        raise ValueError(
            f"max_chunk_size must be at least 1, got {max_chunk_size}"
        )


def min_chunk_size(max_chunk_size: int) -> int:
    """Size under which chunks get merged (smaller for small limits)."""
    return min(MIN_CHUNK_SIZE, max_chunk_size // 4)


def make_chunk(file_path: str, start: int, text: str, title: str) -> Chunck:
    """Build the chunk of `text` that starts at offset `start` of the file."""
    return Chunck(
        file_path=file_path,
        first_character_index=start,
        last_character_index=start + len(text),
        size=len(text),
        text=text,
        title=title,
    )


def sub_chunk(parent: Chunck, offset: int, text: str) -> Chunck:
    """Chunk taken inside `parent`, `offset` characters after its start."""
    return make_chunk(
        parent.file_path,
        parent.first_character_index + offset,
        text,
        parent.title,
    )


def protected_block(
    offset: int,
    blocks: Sequence[Block],
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> Block | None:
    """The block that a cut at `offset` would split, if it fits in one chunk.

    None means the cut is allowed: it is outside any block, or inside one
    bigger than `max_chunk_size`, which has to be cut anyway.
    """
    for start, end in blocks:
        if start < offset < end and end - start <= max_chunk_size:
            return start, end
    return None


def split_at(
    text: str,
    file_path: str,
    boundaries: list[tuple[int, str]],
    default_title: str,
) -> list[Chunck]:
    """One chunk per `(offset, title)` boundary, up to the next one.

    The text before the first boundary gets `default_title`.
    """
    if not boundaries or boundaries[0][0] > 0:
        boundaries = [(0, default_title)] + boundaries
    chunks: list[Chunck] = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        if end > start:
            chunks.append(make_chunk(file_path, start, text[start:end], title))
    return chunks


def split_blankline(
    chunk: Chunck,
    blocks: Sequence[Block] = (),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunck]:
    """One chunk per paragraph, blank lines staying with the one before.

    A blank line inside a block that fits in one chunk is not a cut point.
    """
    text = chunk.text
    base = chunk.first_character_index
    parts: list[Chunck] = []
    start = 0
    for match in BLANK_LINE_RE.finditer(text):
        cut = match.end()
        if (
            cut < len(text)
            and protected_block(base + cut, blocks, max_chunk_size) is None
        ):
            parts.append(sub_chunk(chunk, start, text[start:cut]))
            start = cut
    parts.append(sub_chunk(chunk, start, text[start:]))
    return parts


def split_limits(
    chunk: Chunck,
    blocks: Sequence[Block] = (),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunck]:
    """Cut into parts of at most `max_chunk_size` characters, evenly.

    A cut goes back to the previous line break when there is one in the second
    half of the part. A cut falling inside a block that fits in one chunk is
    moved before the block (or after it if the block starts the part).
    """
    check_max_chunk_size(max_chunk_size)
    text = chunk.text
    base = chunk.first_character_index
    parts: list[Chunck] = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        step = math.ceil(remaining / math.ceil(remaining / max_chunk_size))
        end = start + step
        if end < len(text):
            line_break = text.rfind("\n", start, end) + 1
            if line_break > start and line_break - start >= step // 2:
                end = line_break
            if block := protected_block(base + end, blocks, max_chunk_size):
                block_start, block_end = block[0] - base, block[1] - base
                end = (
                    block_start
                    if block_start > start
                    else min(block_end, len(text))
                )
        parts.append(sub_chunk(chunk, start, text[start:end]))
        start = end
    return parts


def fusion_chuncks(
    chunks: list[Chunck],
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunck]:
    """Merge a too small chunk with its neighbour, up to `max_chunk_size`.

    Only meant for the chunks of one section: a chunk never spans two titles.
    """
    smallest = min_chunk_size(max_chunk_size)
    merged: list[Chunck] = []
    for chunk in chunks:
        if (
            merged
            and min(merged[-1].size, chunk.size) < smallest
            and merged[-1].size + chunk.size <= max_chunk_size
        ):
            prev = merged[-1]
            merged[-1] = make_chunk(
                prev.file_path,
                prev.first_character_index,
                prev.text + chunk.text,
                prev.title,
            )
        else:
            merged.append(chunk)
    return merged


def refine(
    sections: list[Chunck],
    blocks: Sequence[Block] = (),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunck]:
    """Cut sections bigger than `max_chunk_size`: by paragraph, then size."""
    check_max_chunk_size(max_chunk_size)
    result: list[Chunck] = []
    for section in sections:
        if section.size <= max_chunk_size:
            result.append(section)
            continue
        parts: list[Chunck] = []
        for part in split_blankline(section, blocks, max_chunk_size):
            if part.size <= max_chunk_size:
                parts.append(part)
            else:
                parts.extend(split_limits(part, blocks, max_chunk_size))
        result.extend(fusion_chuncks(parts, max_chunk_size))
    return result
