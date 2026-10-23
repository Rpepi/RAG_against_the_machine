import pytest
from helpers import code_block
from src.config import DEFAULT_MAX_CHUNK_SIZE, MAX_CHUNK_SIZE
from src.chunking.common import (
    fusion_chuncks,
    make_chunk,
    protected_block,
    read_source,
    refine,
    split_at,
    split_blankline,
    split_limits,
)


def chunk(text: str, title: str = "T", start: int = 0):
    return make_chunk("data/raw/x", start, text, title)


def contiguous(sizes: list[int], title: str = "T"):
    """Chunks of the given sizes, one right after the other."""
    chunks, start = [], 0
    for i, size in enumerate(sizes):
        chunks.append(chunk("x" * size, f"{title}{i}", start))
        start += size
    return chunks


def joined(parts) -> str:
    return "".join(part.text for part in parts)


class TestMakeChunk:
    def test_fills_every_field(self):
        c = make_chunk("data/raw/a.md", 10, "hello", "Title")
        assert (
            c.file_path,
            c.first_character_index,
            c.last_character_index,
        ) == ("data/raw/a.md", 10, 15)
        assert (c.size, c.text, c.title) == (5, "hello", "Title")


class TestReadSource:
    def test_file_path_is_relative_to_the_project_with_raw_prefix(
        self, project
    ):
        path = project("docs/a.md", "hello")
        text, file_path = read_source(path)
        assert (text, file_path) == ("hello", "data/raw/docs/a.md")

    def test_keeps_windows_line_endings(self, project):
        text, _ = read_source(project("a.txt", "a\r\nb\r\n"))
        assert text == "a\r\nb\r\n"

    def test_path_already_under_data_raw_is_not_prefixed_twice(self, project):
        _, file_path = read_source(
            project("data/raw/vllm-0.10.1/docs/a.md", "x")
        )
        assert file_path == "data/raw/vllm-0.10.1/docs/a.md"

    def test_data_raw_prefix_only_counts_at_the_root(self, project):
        _, file_path = read_source(project("docs/data/raw/a.md", "x"))
        assert file_path == "data/raw/docs/data/raw/a.md"

    def test_accepts_a_string_path(self, project):
        _, file_path = read_source(str(project("a.txt", "x")))
        assert file_path == "data/raw/a.txt"


class TestProtectedBlock:
    def test_cut_strictly_inside_a_small_block_is_protected(self):
        assert protected_block(50, [(10, 100)]) == (10, 100)

    def test_cut_on_the_edges_is_allowed(self):
        assert protected_block(10, [(10, 100)]) is None
        assert protected_block(100, [(10, 100)]) is None

    def test_cut_outside_is_allowed(self):
        assert protected_block(5, [(10, 100)]) is None

    def test_block_bigger_than_max_can_be_cut(self):
        assert protected_block(500, [(0, MAX_CHUNK_SIZE + 1)]) is None

    def test_block_of_exactly_max_is_protected(self):
        assert protected_block(500, [(0, MAX_CHUNK_SIZE)]) == (
            0,
            MAX_CHUNK_SIZE,
        )


class TestSplitAt:
    def test_one_chunk_per_boundary(self):
        parts = split_at(
            "abcdef", "data/raw/x", [(0, "A"), (2, "B"), (4, "C")], "D"
        )
        assert [(p.title, p.text) for p in parts] == [
            ("A", "ab"),
            ("B", "cd"),
            ("C", "ef"),
        ]

    def test_text_before_the_first_boundary_gets_the_default_title(self):
        parts = split_at("abcdef", "data/raw/x", [(2, "B")], "D")
        assert [(p.title, p.text) for p in parts] == [
            ("D", "ab"),
            ("B", "cdef"),
        ]

    def test_no_boundary_is_one_chunk(self):
        parts = split_at("abc", "data/raw/x", [], "D")
        assert [(p.title, p.text) for p in parts] == [("D", "abc")]

    def test_empty_sections_are_dropped(self):
        parts = split_at("abc", "data/raw/x", [(0, "A"), (0, "B")], "D")
        assert [(p.title, p.text) for p in parts] == [("B", "abc")]

    def test_indexes_are_the_offsets_in_the_text(self):
        parts = split_at("abcdef", "data/raw/x", [(0, "A"), (2, "B")], "D")
        assert [
            (p.first_character_index, p.last_character_index) for p in parts
        ] == [(0, 2), (2, 6)]


class TestSplitBlankline:
    def test_cuts_after_each_blank_line(self):
        parts = split_blankline(chunk("a\n\nb\n\nc"))
        assert [p.text for p in parts] == ["a\n\n", "b\n\n", "c"]

    def test_blank_line_with_spaces_counts(self):
        parts = split_blankline(chunk("a\n  \n\t\nb"))
        assert [p.text for p in parts] == ["a\n  \n\t\n", "b"]

    def test_several_blank_lines_stay_with_the_paragraph_before(self):
        parts = split_blankline(chunk("a\n\n\n\nb"))
        assert [p.text for p in parts] == ["a\n\n\n\n", "b"]

    def test_windows_line_endings(self):
        parts = split_blankline(chunk("a\r\n\r\nb"))
        assert [p.text for p in parts] == ["a\r\n\r\n", "b"]

    def test_no_blank_line_is_one_part(self):
        assert [p.text for p in split_blankline(chunk("a\nb\nc"))] == [
            "a\nb\nc"
        ]

    def test_text_ending_with_a_blank_line_makes_no_empty_part(self):
        assert [p.text for p in split_blankline(chunk("a\n\n"))] == ["a\n\n"]

    def test_parts_keep_title_and_absolute_offsets(self):
        parts = split_blankline(chunk("a\n\nb", title="Sec", start=100))
        assert [
            (p.title, p.first_character_index, p.last_character_index)
            for p in parts
        ] == [
            ("Sec", 100, 103),
            ("Sec", 103, 104),
        ]

    def test_blank_line_inside_a_small_block_is_not_a_cut(self):
        block = "```\ncode1\n\ncode2\n```\n"
        text = "p1\n\n" + block + "\np2"
        parts = split_blankline(chunk(text), [(4, 4 + len(block))])
        assert [p.text for p in parts] == ["p1\n\n", block + "\n", "p2"]

    def test_blank_line_inside_a_big_block_is_a_cut(self):
        block = "```\n" + ("l\n" * 400 + "\n") * 3 + "```\n"
        text = "p1\n\n" + block + "\np2"
        parts = split_blankline(chunk(text), [(4, 4 + len(block))])
        assert len(parts) > 3
        assert joined(parts) == text
        assert any(4 < p.first_character_index < 4 + len(block) for p in parts)


class TestSplitLimits:
    def test_parts_are_even(self):
        assert [
            p.size
            for p in split_limits(chunk("a" * 3001), max_chunk_size=1500)
        ] == [1001, 1000, 1000]

    def test_small_text_is_not_cut(self):
        assert [p.size for p in split_limits(chunk("a" * 100))] == [100]

    def test_hard_cut_when_there_is_no_line_break(self):
        parts = split_limits(chunk("a" * 4000))
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)
        assert joined(parts) == "a" * 4000

    def test_cut_goes_back_to_the_previous_line_break(self):
        text = "abcdefghijkl\n" * 201
        parts = split_limits(chunk(text))
        assert joined(parts) == text
        assert all(p.text.endswith("\n") for p in parts)
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)

    def test_line_break_too_early_is_ignored(self):
        text = "ab\n" + "c" * 3000
        parts = split_limits(chunk(text))
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)
        assert joined(parts) == text
        assert parts[0].size > 100

    def test_cut_moves_before_a_small_block(self):
        text = "a" * 900 + "\n" + code_block(500) + "b" * 600
        parts = split_limits(chunk(text), [(901, 1401)])
        assert [p.size for p in parts] == [901, 1100]
        assert sum(code_block(500) in p.text for p in parts) == 1

    def test_cut_moves_after_a_small_block_that_starts_the_part(self):
        text = code_block(1400) + "b" * 1000
        parts = split_limits(chunk(text), [(0, 1400)])
        assert [p.size for p in parts] == [1400, 1000]

    def test_big_block_is_cut(self):
        text = "```\n" + "line\n" * 400 + "```\n"
        parts = split_limits(chunk(text), [(0, len(text))])
        assert len(parts) == 2
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)
        assert joined(parts) == text

    def test_offsets_are_absolute(self):
        parts = split_limits(chunk("a" * 3000, start=500))
        assert [p.first_character_index for p in parts] == [500, 2000]
        assert parts[-1].last_character_index == 3500


class TestFusion:
    def test_small_neighbours_are_merged_up_to_the_max(self):
        merged = fusion_chuncks(contiguous([100, 100, 100]))
        assert [(m.size, m.title) for m in merged] == [(300, "T0")]

    def test_merged_chunk_covers_the_same_range(self):
        merged = fusion_chuncks(contiguous([100, 100]))
        assert (
            merged[0].first_character_index,
            merged[0].last_character_index,
        ) == (0, 200)
        assert merged[0].text == "x" * 200

    def test_never_exceeds_the_max(self):
        merged = fusion_chuncks(contiguous([1400, 200]), max_chunk_size=1500)
        assert [m.size for m in merged] == [1400, 200]

    def test_exactly_the_max_is_allowed(self):
        assert [
            m.size
            for m in fusion_chuncks(
                contiguous([100, 1400]), max_chunk_size=1500
            )
        ] == [1500]

    def test_chunks_at_or_above_min_are_left_alone(self):
        assert [m.size for m in fusion_chuncks(contiguous([400, 400]))] == [
            400,
            400,
        ]

    def test_empty_list(self):
        assert fusion_chuncks([]) == []


class TestRefine:
    def test_small_sections_are_kept_and_never_merged(self):
        sections = [chunk("a" * 100, "A", 0), chunk("b" * 100, "B", 100)]
        assert [(c.title, c.size) for c in refine(sections)] == [
            ("A", 100),
            ("B", 100),
        ]

    def test_big_section_is_cut_and_keeps_its_title(self):
        text = ("x" * 700 + "\n\n") * 5
        parts = refine([chunk(text, "Sec")])
        assert len(parts) > 1
        assert {p.title for p in parts} == {"Sec"}
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)
        assert joined(parts) == text

    def test_small_paragraphs_of_a_big_section_are_packed(self):
        text = ("x" * 98 + "\n\n") * 40
        parts = refine([chunk(text)], max_chunk_size=1500)
        assert joined(parts) == text
        assert len(parts) == 3
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)

    def test_paragraph_bigger_than_max_is_cut_by_size(self):
        parts = refine([chunk("x" * 5000)])
        assert all(p.size <= MAX_CHUNK_SIZE for p in parts)
        assert joined(parts) == "x" * 5000

    def test_sections_are_processed_independently(self):
        sections = [
            chunk("a" * 100, "A", 0),
            chunk(("b" * 700 + "\n\n") * 4, "B", 100),
        ]
        titles = [c.title for c in refine(sections)]
        assert titles[0] == "A"
        assert set(titles[1:]) == {"B"}


@pytest.mark.parametrize("size", [1, 1499, 1500, 1501, 3000, 3001, 10_000])
def test_split_limits_always_tiles_and_respects_the_max(size):
    text = "y" * size
    parts = split_limits(chunk(text))
    assert joined(parts) == text
    assert all(0 < p.size <= MAX_CHUNK_SIZE for p in parts)


@pytest.mark.parametrize("size", [1, 5, 100, 1999, 2000, 2001, 4001])
def test_default_max_is_2000(size):
    parts = split_limits(chunk("y" * size))
    assert DEFAULT_MAX_CHUNK_SIZE == 2000
    assert all(0 < p.size <= 2000 for p in parts)


@pytest.mark.parametrize("max_size", [1, 10, 300, 1000, 2000, 5000])
def test_refine_respects_any_max_chunk_size(max_size):
    text = ("word " * 30 + "\n\n") * 50 + "z" * 3000
    parts = refine([chunk(text)], max_chunk_size=max_size)
    assert joined(parts) == text
    assert all(0 < p.size <= max_size for p in parts)


@pytest.mark.parametrize("bad", [0, -1])
def test_max_chunk_size_must_be_positive(bad):
    with pytest.raises(ValueError):
        refine([chunk("abc" * 1000)], max_chunk_size=bad)
    with pytest.raises(ValueError):
        split_limits(chunk("abc"), max_chunk_size=bad)
