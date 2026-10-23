import pytest
from helpers import VLLM, assert_valid, code_block, cut_offsets, read_raw
from src.config import MAX_CHUNK_SIZE
from src.chunking.markdown_chunking import (
    code_blocks,
    markdown_chunker,
    split_title,
)

FP = "data/raw/x.md"


def paragraphs(count: int, size: int = 400) -> str:
    return "".join(f"{'p' * (size - 2)}\n\n" for _ in range(count))


class TestCodeBlocks:
    def test_span_covers_the_fences(self):
        assert code_blocks("a\n```py\ncode\n```\nb\n") == [(2, 17)]

    def test_no_block(self):
        assert code_blocks("just text\n# not code\n") == []

    def test_indented_fence(self):
        text = "- item\n\n  ```\n  code\n  ```\n"
        ((start, end),) = code_blocks(text)
        assert text[start:end] == "  ```\n  code\n  ```\n"

    def test_several_blocks(self):
        text = "```\na\n```\ntext\n```\nb\n```\n"
        assert len(code_blocks(text)) == 2

    def test_unclosed_fence_runs_to_the_end(self):
        assert code_blocks("a\n```\ncode") == [(2, 10)]


class TestSplitTitle:
    def test_one_section_per_title(self):
        text = "# A\nx\n## B\ny\n### C\nz\n"
        parts = split_title(text, FP, [])
        assert [(p.title, p.text) for p in parts] == [
            ("A", "# A\nx\n"),
            ("B", "## B\ny\n"),
            ("C", "### C\nz\n"),
        ]

    def test_text_before_the_first_title_has_an_empty_title(self):
        parts = split_title("intro\n# A\nx\n", FP, [])
        assert [(p.title, p.text) for p in parts] == [
            ("", "intro\n"),
            ("A", "# A\nx\n"),
        ]

    def test_no_title_is_one_section(self):
        parts = split_title("just\ntext\n", FP, [])
        assert [(p.title, p.text) for p in parts] == [("", "just\ntext\n")]

    def test_fourth_level_is_not_a_boundary(self):
        parts = split_title("# A\n#### deep\nx\n", FP, [])
        assert len(parts) == 1

    def test_hash_without_space_is_not_a_title(self):
        assert len(split_title("# A\n#hashtag\n", FP, [])) == 1

    def test_hash_inside_a_code_block_is_not_a_title(self):
        text = "# A\n```\n# comment\n```\n"
        parts = split_title(text, FP, code_blocks(text))
        assert [p.title for p in parts] == ["A"]

    def test_title_text_is_stripped(self):
        parts = split_title("#   Spaced title   \nx\n", FP, [])
        assert parts[0].title == "Spaced title"

    def test_windows_line_endings(self):
        text = "# A\r\nx\r\n## B\r\ny\r\n"
        parts = split_title(text, FP, [])
        assert [p.title for p in parts] == ["A", "B"]
        assert "".join(p.text for p in parts) == text


class TestMarkdownChunker:
    def test_empty_file(self, project):
        assert markdown_chunker(project("a.md", "")) == []

    def test_blank_file(self, project):
        assert markdown_chunker(project("a.md", "\n  \n")) == []

    def test_file_path_has_the_raw_prefix(self, project):
        chunks = markdown_chunker(project("docs/a.md", "# A\nx\n"))
        assert chunks[0].file_path == "data/raw/docs/a.md"

    def test_accepts_a_string_path(self, project):
        assert (
            markdown_chunker(str(project("a.md", "# A\nx\n")))[0].title == "A"
        )

    def test_small_file_is_one_chunk_per_title(self, project):
        text = "# A\nx\n\n## B\ny\n"
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        assert [c.title for c in chunks] == ["A", "B"]

    def test_small_sections_are_not_merged(self, project):
        text = "# A\nx\n\n## B\ny\n\n## C\nz\n"
        chunks = markdown_chunker(project("a.md", text))
        assert [c.title for c in chunks] == ["A", "B", "C"]
        assert all(c.size < 300 for c in chunks)

    def test_big_section_is_cut_and_keeps_its_title(self, project):
        text = "# Big\n\n" + paragraphs(10)
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        assert len(chunks) >= 3
        assert {c.title for c in chunks} == {"Big"}

    def test_a_chunk_never_spans_two_titles(self, project):
        text = (
            "# A\n"
            + paragraphs(6)
            + "## B\n"
            + paragraphs(6, 100)
            + "## C\nend\n"
        )
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        for chunk in chunks:
            assert "\n## " not in chunk.text and "\n# " not in chunk.text
        assert [c.title for c in chunks if c.title == "C"] == ["C"]
        assert chunks[-1].text.startswith("## C")

    def test_each_title_starts_a_chunk(self, project):
        text = "# A\n" + paragraphs(6) + "## B\n" + paragraphs(6)
        chunks = markdown_chunker(project("a.md", text))
        starts = {text.index("## B")}
        assert starts <= set(cut_offsets(chunks))

    def test_small_code_block_is_never_cut(self, project):
        block = (
            "```python\n" + ("c" * 348 + "\n\n") * 3 + "```\n"
        )  # paragraphs too big to be merged back
        assert 900 < len(block) <= MAX_CHUNK_SIZE
        text = (
            "# A\n\n" + paragraphs(3, 600) + block + "\n" + paragraphs(3, 600)
        )
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        assert sum(block in c.text for c in chunks) == 1

    def test_small_code_block_is_never_cut_by_a_size_cut(self, project):
        block = code_block(500)
        text = "# A\n\n" + "a" * 900 + "\n" + block + "b" * 600
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        assert sum(block in c.text for c in chunks) == 1

    def test_code_block_bigger_than_the_max_is_cut(self, project):
        block = "```\n" + ("line of code\n" * 30 + "\n") * 12 + "```\n"
        assert len(block) > MAX_CHUNK_SIZE
        text = "# A\n\n" + block
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        assert len(chunks) >= 2
        assert any(block not in c.text for c in chunks)

    def test_windows_line_endings_keep_the_indexes(self, project):
        text = (
            "# Title\r\n"
            + "word " * 20
            + "\r\n\r\n"
            + ("para " * 60 + "\r\n\r\n") * 8
        )
        chunks = markdown_chunker(project("a.md", text))
        assert_valid(chunks, text)
        assert chunks[0].title == "Title"


@pytest.mark.skipif(
    not (VLLM / "docs").exists(), reason="vLLM sources not present"
)
class TestRealDocs:
    FILES = (
        sorted((VLLM / "docs").rglob("*.md"))
        if (VLLM / "docs").exists()
        else []
    )

    def test_there_are_files(self):
        assert len(self.FILES) > 100

    def test_every_doc_is_chunked_correctly(self):
        problems = []
        for path in self.FILES:
            text = read_raw(path)
            chunks = markdown_chunker(path)
            if not text.strip():
                assert chunks == []
                continue
            try:
                assert_valid(chunks, text)
            except AssertionError as error:
                problems.append(f"{path.relative_to(VLLM)}: {error}")
        assert not problems, "\n".join(problems[:10])

    def test_no_chunk_spans_two_titles(self):
        problems = []
        for path in self.FILES:
            text = read_raw(path)
            if not text.strip():
                continue
            sections = split_title(text, FP, code_blocks(text))
            bounds = [s.first_character_index for s in sections[1:]]
            for chunk in markdown_chunker(path):
                inside = [
                    b
                    for b in bounds
                    if chunk.first_character_index
                    < b
                    < chunk.last_character_index
                ]
                if inside:
                    problems.append(
                        f"{path.relative_to(VLLM)} "
                        f"chunk {chunk.first_character_index}"
                    )
        assert not problems, "\n".join(problems[:10])

    def test_no_small_code_block_is_cut(self):
        problems = []
        for path in self.FILES:
            text = read_raw(path)
            if not text.strip():
                continue
            blocks = [
                (s, e) for s, e in code_blocks(text) if e - s <= MAX_CHUNK_SIZE
            ]
            for cut in cut_offsets(markdown_chunker(path)):
                if any(s < cut < e for s, e in blocks):
                    problems.append(f"{path.relative_to(VLLM)} cut at {cut}")
        assert not problems, "\n".join(problems[:10])
