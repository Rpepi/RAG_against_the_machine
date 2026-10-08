import ast
import warnings
from pathlib import Path
from src.indexing.chunking.models import Chunck
from src.indexing.chunking.common import (
    DEFAULT_MAX_CHUNK_SIZE,
    read_source,
    refine,
    split_at,
)

DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def parse(text: str) -> ast.Module | None:
    """The syntax tree, or None if the file is not valid Python."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return None


def definition_span(node: ast.AST, lines: list[str], offsets: list[int]) -> tuple[int, int]:
    """(start, end) offsets of a definition.

    It starts at its decorators and at the comment lines right above it, and
    ends after the blank lines that follow it.
    """
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    while (first > 1
            and lines[first - 2].lstrip().startswith("#")
            and len(lines[first - 2]) - len(lines[first - 2].lstrip()) == node.col_offset):
        first -= 1
    last = node.end_lineno
    while last < len(lines) and not lines[last].strip():
        last += 1
    return offsets[first - 1], min(offsets[last], offsets[-1] - 1)


def collect_boundaries(
    body: list[ast.stmt],
    scope: str,
    gap_title: str,
    lines: list[str],
    offsets: list[int],
    boundaries: dict[int, str],
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> None:
    """Fill `boundaries` with (offset -> title) for the definitions of `body`.

    A class bigger than `max_chunk_size` is opened up: its methods get their own
    boundaries and the rest of the class keeps the class name.
    """
    for node in body:
        if not isinstance(node, DEFINITIONS):
            continue
        start, end = definition_span(node, lines, offsets)
        name = scope + node.name
        boundaries[start] = name
        if isinstance(node, ast.ClassDef) and end - start > max_chunk_size:
            collect_boundaries(
                node.body, name + ".", name, lines, offsets, boundaries, max_chunk_size
            )
        boundaries[end] = gap_title


def split_definitions(
    text: str,
    file_path: str,
    title: str,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[Chunck]:
    """One chunk per function or class, the code between them being chunks of its own.

    `title` names that code between definitions. A file that does not parse
    is one single section.
    """
    tree = parse(text)
    boundaries: dict[int, str] = {}
    if tree is not None:
        lines = text.split("\n")
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line) + 1)
        collect_boundaries(tree.body, "", title, lines, offsets, boundaries, max_chunk_size)
    return split_at(text, file_path, sorted(boundaries.items()), title)


def python_chunker(path: Path, max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE) -> list[Chunck]:
    text, file_path = read_source(path)
    if not text.strip():
        return []
    return refine(
        split_definitions(text, file_path, Path(path).name, max_chunk_size),
        max_chunk_size=max_chunk_size,
    )


if __name__ == "__main__":
    from src.indexing.chunking.common import PROJECT_ROOT

    for chunk in python_chunker(PROJECT_ROOT / "data" / "raw" / "vllm-0.10.1" / "vllm" / "plugins" / "__init__.py"):
        print(f"[{chunk.first_character_index}:{chunk.last_character_index}] {chunk.title!r}")
        print(chunk.text)
