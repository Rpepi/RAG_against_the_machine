import pytest
from helpers import VLLM, assert_valid, cut_offsets, read_raw
from src.indexing.chunking.common import MAX_CHUNK_SIZE
from src.indexing.chunking.text_chunking import text_chunker


class TestTextChunker:
    def test_empty_file(self, project):
        assert text_chunker(project("a.txt", "")) == []

    def test_blank_file(self, project):
        assert text_chunker(project("a.txt", "\n\n  \n")) == []

    def test_small_file_is_one_chunk_titled_with_the_file_name(self, project):
        text = "torch==2.7.1\nnumpy\n"
        chunks = text_chunker(project("requirements/cuda.txt", text))
        assert_valid(chunks, text)
        assert len(chunks) == 1
        assert chunks[0].title == "cuda.txt"
        assert chunks[0].file_path == "data/raw/requirements/cuda.txt"

    def test_accepts_a_string_path(self, project):
        assert len(text_chunker(str(project("a.txt", "x")))) == 1

    def test_big_file_is_cut_by_paragraph(self, project):
        text = ("word " * 99 + "\n\n") * 10
        chunks = text_chunker(project("a.txt", text))
        assert_valid(chunks, text)
        assert len(chunks) >= 3
        assert {c.title for c in chunks} == {"a.txt"}
        assert all(c.text.endswith("\n\n") for c in chunks)

    def test_small_paragraphs_are_packed_together(self, project):
        text = ("word " * 19 + "\n\n") * 40
        chunks = text_chunker(project("a.txt", text), max_chunk_size=1500)
        assert_valid(chunks, text, 1500)
        assert len(chunks) == 3

    def test_list_without_blank_line_is_cut_between_lines(self, project):
        text = "".join(f"package{i:03d}==1.2.3\n" for i in range(200))
        chunks = text_chunker(project("requirements.txt", text))
        assert_valid(chunks, text)
        assert len(chunks) >= 2
        assert all(c.text.endswith("\n") for c in chunks)
        assert all(text[cut - 1] == "\n" for cut in cut_offsets(chunks))

    def test_line_longer_than_the_max_is_cut_anyway(self, project):
        text = "a" * 4000
        chunks = text_chunker(project("a.txt", text), max_chunk_size=1500)
        assert_valid(chunks, text, 1500)
        assert len(chunks) == 3

    def test_windows_line_endings_keep_the_indexes(self, project):
        text = "".join(f"line {i}\r\n" for i in range(400))
        chunks = text_chunker(project("a.txt", text))
        assert_valid(chunks, text)
        assert len(chunks) >= 2


@pytest.mark.skipif(not VLLM.exists(), reason="vLLM sources not present")
class TestRealTextFiles:
    FILES = sorted(p for p in VLLM.rglob("*.txt") if p.is_file()) if VLLM.exists() else []

    def test_there_are_files(self):
        assert len(self.FILES) >= 20

    def test_every_text_file_is_chunked_correctly(self):
        problems = []
        for path in self.FILES:
            text = read_raw(path)
            chunks = text_chunker(path)
            if not text.strip():
                assert chunks == []
                continue
            try:
                assert_valid(chunks, text)
                assert {c.title for c in chunks} == {path.name}
            except AssertionError as error:
                problems.append(f"{path.relative_to(VLLM)}: {error}")
        assert not problems, "\n".join(problems[:10])

    def test_files_made_of_lines_are_only_cut_between_lines(self):
        checked = 0
        for path in self.FILES:
            text = read_raw(path)
            if len(text) <= MAX_CHUNK_SIZE or text.count("\n") < len(text) / 200:
                continue
            checked += 1
            cuts = cut_offsets(text_chunker(path))
            assert cuts
            assert all(text[cut - 1] == "\n" for cut in cuts), path.name
        assert checked >= 5

    def test_file_without_any_line_break_is_still_cut(self):
        path = VLLM / "tests" / "prompts" / "summary.txt"
        text = read_raw(path)
        assert "\n" not in text
        chunks = text_chunker(path)
        assert_valid(chunks, text)
        assert len(chunks) > 1
