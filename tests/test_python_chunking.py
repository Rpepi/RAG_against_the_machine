import ast

import pytest
from helpers import VLLM, assert_valid, read_raw
from src.indexing.chunking.python_chunking import (
    parse,
    python_chunker,
    split_definitions,
)

SOURCE = '''\
"""Module doc."""
import os

X = 1


# helper for foo
@decorator
def foo(a):
    return a


async def bar():
    pass


class Small:
    def m(self):
        return 1
'''


def chunk_source(project, source: str, name: str = "mod.py"):
    return python_chunker(project(name, source))


def method(name: str, size: int) -> str:
    return f"    def {name}(self):\n        return '{'x' * size}'\n\n"


class TestParse:
    def test_valid_source(self):
        assert parse("x = 1\n") is not None

    def test_invalid_source(self):
        assert parse("def broken(:\n") is None

    def test_null_byte(self):
        assert parse("x = 1\0") is None

    def test_invalid_escape_does_not_warn(self, recwarn):
        assert parse("x = '\\d'\n") is not None
        assert len(recwarn) == 0


class TestPythonChunker:
    def test_empty_file(self, project):
        assert chunk_source(project, "") == []

    def test_blank_file(self, project):
        assert chunk_source(project, "\n\n") == []

    def test_one_chunk_per_definition(self, project):
        chunks = chunk_source(project, SOURCE)
        assert_valid(chunks, SOURCE)
        assert [c.title for c in chunks] == ["mod.py", "foo", "bar", "Small"]

    def test_code_between_definitions_is_titled_with_the_file_name(self, project):
        chunks = chunk_source(project, SOURCE)
        assert chunks[0].text == '"""Module doc."""\nimport os\n\nX = 1\n\n\n'
        assert chunks[0].title == "mod.py"

    def test_decorators_and_comment_above_belong_to_the_function(self, project):
        chunk = chunk_source(project, SOURCE)[1]
        assert chunk.text.startswith("# helper for foo\n@decorator\ndef foo(a):")

    def test_blank_lines_after_a_definition_stay_with_it(self, project):
        chunk = chunk_source(project, SOURCE)[1]
        assert chunk.text.endswith("return a\n\n\n")

    def test_async_function(self, project):
        chunk = chunk_source(project, SOURCE)[2]
        assert chunk.text.startswith("async def bar")

    def test_small_class_is_one_chunk(self, project):
        chunk = chunk_source(project, SOURCE)[3]
        assert chunk.text == "class Small:\n    def m(self):\n        return 1\n"

    def test_file_path_and_indexes(self, project):
        chunks = chunk_source(project, SOURCE, "vllm/mod.py")
        assert chunks[0].file_path == "data/raw/vllm/mod.py"
        assert chunks[1].first_character_index == SOURCE.index("# helper for foo")

    def test_comment_indented_deeper_is_not_pulled_into_the_next_function(self, project):
        source = "def a():\n    pass\n    # trailing\ndef b():\n    pass\n"
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["a", "mod.py", "b"]
        assert chunks[1].text == "    # trailing\n"

    def test_comment_separated_by_a_blank_line_is_not_pulled(self, project):
        source = "def a():\n    pass\n\n# alone\n\ndef b():\n    pass\n"
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["a", "mod.py", "b"]

    def test_definitions_without_blank_line_between(self, project):
        source = "def a():\n    pass\ndef b():\n    pass\n"
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["a", "b"]

    def test_small_definitions_are_not_merged(self, project):
        chunks = chunk_source(project, SOURCE)
        assert all(c.size < 300 for c in chunks)
        assert len(chunks) == 4

    def test_class_bigger_than_the_max_is_opened_up_by_method(self, project):
        source = "class Big:\n    '''doc'''\n\n" + method("m1", 700) + method("m2", 700) + method("m3", 700)
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["Big", "Big.m1", "Big.m2", "Big.m3"]
        assert chunks[0].text == "class Big:\n    '''doc'''\n\n"

    def test_code_after_a_method_keeps_the_class_name(self, project):
        source = "class Big:\n" + method("m1", 700) + "    attr = 1\n\n" + method("m2", 800)
        chunks = python_chunker(project("mod.py", source), max_chunk_size=1500)
        assert_valid(chunks, source, 1500)
        assert [c.title for c in chunks] == ["Big", "Big.m1", "Big", "Big.m2"]

    def test_nested_class_names_are_dotted(self, project):
        inner = "    class Inner:\n" + "".join(f"        def m{i}(self):\n            return '{'x' * 500}'\n\n" for i in range(4))
        source = "class Outer:\n" + inner
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks][:2] == ["Outer", "Outer.Inner"]
        assert "Outer.Inner.m0" in [c.title for c in chunks]

    def test_big_function_is_cut_and_keeps_its_name(self, project):
        body = "".join(f"    step_{i} = compute({i})\n    other_{i} = {'v' * 150}\n\n" for i in range(20))
        source = "def big():\n" + body
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert len(chunks) >= 2
        assert {c.title for c in chunks} == {"big"}

    def test_big_module_level_code_is_cut(self, project):
        source = "".join(f"CONSTANT_{i} = '{'v' * 120}'\n" for i in range(40))
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert len(chunks) >= 3
        assert {c.title for c in chunks} == {"mod.py"}

    def test_invalid_python_is_still_chunked(self, project):
        source = "def broken(:\n    pass\n\n" * 200
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert {c.title for c in chunks} == {"mod.py"}

    def test_file_without_definition(self, project):
        source = "import os\nX = 1\n"
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["mod.py"]

    def test_windows_line_endings_keep_the_indexes(self, project):
        source = SOURCE.replace("\n", "\r\n")
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["mod.py", "foo", "bar", "Small"]

    def test_non_ascii_characters_keep_the_indexes(self, project):
        source = "# é à ü 日本語\nNAME = 'héllo'\n\n\ndef f():\n    return 'ñ'\n"
        chunks = chunk_source(project, source)
        assert_valid(chunks, source)
        assert [c.title for c in chunks] == ["mod.py", "f"]

    def test_accepts_a_string_path(self, project):
        assert chunk_source(project, "def a():\n    pass\n")[0].title == "a"


class TestSplitDefinitions:
    def test_returns_sections_before_any_size_cut(self):
        sections = split_definitions(SOURCE, "data/raw/mod.py", "mod.py")
        assert [s.title for s in sections] == ["mod.py", "foo", "bar", "Small"]
        assert "".join(s.text for s in sections) == SOURCE

    def test_definition_ending_the_file_without_newline(self):
        sections = split_definitions("def a():\n    pass", "data/raw/mod.py", "mod.py")
        assert [(s.title, s.text) for s in sections] == [("a", "def a():\n    pass")]


@pytest.mark.skipif(not (VLLM / "vllm").exists(), reason="vLLM sources not present")
class TestRealPythonFiles:
    FILES = sorted(p for p in VLLM.rglob("*.py") if p.is_file()) if (VLLM / "vllm").exists() else []

    def test_there_are_files(self):
        assert len(self.FILES) > 1000

    def test_every_python_file_is_chunked_correctly(self):
        problems = []
        for path in self.FILES:
            text = read_raw(path)
            chunks = python_chunker(path)
            if not text.strip():
                assert chunks == []
                continue
            try:
                assert_valid(chunks, text)
                assert all(c.title for c in chunks)
            except AssertionError as error:
                problems.append(f"{path.relative_to(VLLM)}: {error}")
        assert not problems, "\n".join(problems[:10])

    def test_a_chunk_never_holds_two_top_level_definitions(self):
        problems = []
        for path in self.FILES:
            text = read_raw(path)
            tree = parse(text)
            if tree is None:
                continue
            offsets = [0]
            for line in text.split("\n"):
                offsets.append(offsets[-1] + len(line) + 1)
            starts = [
                offsets[node.lineno - 1] for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            for chunk in python_chunker(path):
                inside = [s for s in starts if chunk.first_character_index <= s < chunk.last_character_index]
                if len(inside) > 1:
                    problems.append(f"{path.relative_to(VLLM)} {chunk.title} holds {len(inside)} definitions")
        assert not problems, "\n".join(problems[:10])
