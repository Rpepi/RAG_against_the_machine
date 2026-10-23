"""Tests of `answer` and `answer_dataset` without loading the real model.

The tokenizer and the model are replaced by fakes that record what they
receive: the tests check our logic (passages, messages, cutting the prompt
off the output, output file), not the quality of Qwen.
"""

import json
from typing import Any

import pytest
import torch
from src.answer import answer as answer_module
from src.answer.answer import PROMPT, answer, answer_dataset
from src.answer.answer import find_answer_sources
from src.config import GENERATION_SETTINGS, MODEL_NAME
from src.data_models.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.utils.load_model import EvaluationError

PROMPT_IDS = [11, 12, 13]  # what the fake tokenizer returns for any prompt


class FakeInputs(dict):
    """What `tokenizer([text], return_tensors="pt")` returns."""

    def __init__(self):
        ids = torch.tensor([PROMPT_IDS])
        super().__init__(input_ids=ids, attention_mask=torch.ones_like(ids))
        self.input_ids = ids

    def to(self, device):
        return self


class FakeTokenizer:
    def __init__(self):
        self.messages = []  # the messages of each apply_chat_template call
        self.template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(messages)
        self.template_kwargs.append(kwargs)
        return "formatted prompt"

    def __call__(self, texts, return_tensors=None):
        assert texts == ["formatted prompt"]
        assert return_tensors == "pt"
        return FakeInputs()

    def decode(self, ids, skip_special_tokens=False):
        assert skip_special_tokens
        return "\n" + " ".join(f"tok{i}" for i in ids) + "\n"


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        n = len(self.calls)
        new_tokens = torch.tensor([[100 + n, 200 + n]])  # differs per call
        return torch.cat([kwargs["input_ids"], new_tokens], dim=1)


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """Fake model, fake tokenizer and a fake project root with 2 files."""
    tokenizer, model = FakeTokenizer(), FakeModel()
    loads: list[tuple[Any, ...]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            loads.append(("tokenizer", name))
            return tokenizer

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append(("model", name, kwargs))
            return model

    monkeypatch.setattr(answer_module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(answer_module, "AutoModelForCausalLM", FakeAutoModel)
    monkeypatch.setattr(answer_module, "PROJECT_ROOT", tmp_path)

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    # non-ASCII before the passage: character and byte offsets differ
    (raw / "a.md").write_text("✅ intro\nLoRA endpoint: /v1/load\n",
                              encoding="utf-8", newline="")
    (raw / "b.py").write_text("x = 1\r\ndef f():\r\n    return 2\r\n",
                              encoding="utf-8", newline="")
    return tokenizer, model, loads


def source(name, first, last):
    return MinimalSource(file_path=f"data/raw/{name}",
                         first_character_index=first,
                         last_character_index=last)


A_PASSAGE = source("a.md", 8, 32)  # "LoRA endpoint: /v1/load\n"
B_PASSAGE = source("b.py", 7, 31)  # "def f():\r\n    return 2\r\n"


class TestFindAnswerSources:
    def test_reads_the_exact_character_range(self, fakes):
        assert find_answer_sources([A_PASSAGE]) == [
            "LoRA endpoint: /v1/load\n"
        ]

    def test_keeps_windows_line_endings(self, fakes):
        assert find_answer_sources([B_PASSAGE]) == [
            "def f():\r\n    return 2\r\n"
        ]

    def test_keeps_the_order_of_the_sources(self, fakes):
        passages = find_answer_sources([B_PASSAGE, A_PASSAGE])
        assert passages[0].startswith("def f")
        assert passages[1].startswith("LoRA")

    def test_missing_file(self, fakes):
        with pytest.raises(FileNotFoundError):
            find_answer_sources([source("nope.md", 0, 5)])


class TestAnswer:
    @pytest.fixture
    def run(self, fakes, monkeypatch, capsys):
        calls = []

        def fake_retrieve(query, k):
            calls.append((query, k))
            return [A_PASSAGE, B_PASSAGE]

        monkeypatch.setattr(answer_module, "retrieve", fake_retrieve)
        answer("Which endpoint loads LoRA?", 2)
        return fakes, calls, capsys.readouterr().out

    def test_searches_with_the_query_and_k(self, run):
        _, calls, _ = run
        assert calls == [("Which endpoint loads LoRA?", 2)]

    def test_loads_the_subject_model_on_cpu_in_float32(self, run):
        (_, _, loads), _, _ = run
        assert ("tokenizer", MODEL_NAME) in loads
        assert ("model", MODEL_NAME, {"dtype": torch.float32}) in loads

    def test_system_message_holds_the_rules(self, run):
        (tokenizer, _, _), _, _ = run
        system, _ = tokenizer.messages[0]
        assert system == {"role": "system", "content": PROMPT}

    def test_user_message_holds_numbered_passages_then_the_question(
        self, run
    ):
        (tokenizer, _, _), _, _ = run
        _, user = tokenizer.messages[0]
        assert user["role"] == "user"
        content = user["content"]
        assert "[1] data/raw/a.md\nLoRA endpoint: /v1/load" in content
        assert "[2] data/raw/b.py\ndef f():" in content
        assert content.index("[2]") < content.index("Which endpoint")

    def test_chat_template_options(self, run):
        (tokenizer, _, _), _, _ = run
        assert tokenizer.template_kwargs[0] == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }

    def test_uses_the_generation_settings(self, run):
        (_, model, _), _, _ = run
        for key, value in GENERATION_SETTINGS.items():
            assert model.calls[0][key] == value

    def test_prints_only_the_new_tokens(self, run):
        _, _, out = run
        assert out == "content: tok101 tok201\n"


def write_results(path, questions, k=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    results = StudentSearchResults(
        k=k,
        search_results=[
            MinimalSearchResults(
                question_id=f"q{i}", question=q, retrieved_sources=sources
            )
            for i, (q, sources) in enumerate(questions, 1)
        ],
    )
    path.write_text(results.model_dump_json(), encoding="utf-8")
    return path


QUESTIONS = [
    ("Which endpoint loads LoRA?", [A_PASSAGE, B_PASSAGE]),
    ("What does f return?", [B_PASSAGE]),
]


class TestAnswerDataset:
    @pytest.fixture
    def run(self, fakes, tmp_path):
        results = write_results(
            tmp_path / "out" / "search" / "dataset_code_public.json",
            QUESTIONS,
        )
        answer_dataset(results, tmp_path / "out" / "answers")
        written = tmp_path / "out" / "answers" / "dataset_code_public.json"
        return fakes, written

    def load(self, path):
        return StudentSearchResultsAndAnswer.model_validate_json(
            path.read_text(encoding="utf-8")
        )

    def test_writes_a_valid_file_named_like_the_input(self, run):
        _, written = run
        assert written.is_file()
        assert len(self.load(written).search_results) == 2

    def test_copies_k_ids_questions_and_sources(self, run):
        _, written = run
        output = self.load(written)
        assert output.k == 10
        first, second = output.search_results
        assert (first.question_id, first.question) == (
            "q1", "Which endpoint loads LoRA?",
        )
        assert first.retrieved_sources == [A_PASSAGE, B_PASSAGE]
        assert second.retrieved_sources == [B_PASSAGE]

    def test_each_answer_is_the_new_tokens_of_its_own_generation(self, run):
        _, written = run
        answers = [a.answer for a in self.load(written).search_results]
        assert answers == ["tok101 tok201", "tok102 tok202"]

    def test_model_is_loaded_once_for_all_questions(self, run):
        (_, _, loads), _ = run
        assert [name for name, *_ in loads].count("model") == 1

    def test_each_question_gets_its_own_prompt(self, run):
        # regression: the prompt was built once, before the loop
        (tokenizer, model, _), _ = run
        assert len(model.calls) == 2
        users = [messages[1]["content"] for messages in tokenizer.messages]
        assert "Which endpoint loads LoRA?" in users[0]
        assert "What does f return?" in users[1]
        assert "[2]" not in users[1]  # only its own single passage

    def test_creates_the_missing_output_folder(self, fakes, tmp_path):
        results = write_results(tmp_path / "r.json", QUESTIONS)
        target = tmp_path / "a" / "b" / "c"
        answer_dataset(results, target)
        assert (target / "r.json").is_file()

    def test_relative_paths_start_at_the_project_root(self, fakes, tmp_path):
        write_results(tmp_path / "data" / "output" / "r.json", QUESTIONS)
        answer_dataset("data/output/r.json", "data/output/answers")
        assert (tmp_path / "data/output/answers/r.json").is_file()

    def test_empty_results_give_an_empty_file(self, fakes, tmp_path):
        results = write_results(tmp_path / "r.json", [])
        answer_dataset(results, tmp_path / "answers")
        output = self.load(tmp_path / "answers" / "r.json")
        assert output.search_results == []

    def test_missing_results_file(self, fakes, tmp_path):
        with pytest.raises(EvaluationError, match="not found"):
            answer_dataset(tmp_path / "nope.json", tmp_path / "answers")

    def test_malformed_results_file(self, fakes, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"k": 10}), encoding="utf-8")
        with pytest.raises(EvaluationError, match="wrong format"):
            answer_dataset(bad, tmp_path / "answers")

    def test_reports_where_it_saved(self, fakes, tmp_path, capsys):
        results = write_results(tmp_path / "r.json", QUESTIONS)
        answer_dataset(results, tmp_path / "answers")
        assert str(tmp_path / "answers" / "r.json") in capsys.readouterr().out
