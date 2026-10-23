import json
import subprocess
import sys

import pytest
from src.config import PROJECT_ROOT
from src.data_models.models import Chunck, RagDataset, UnansweredQuestion
from src.retrieval.bm25 import BM25Index


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def index_path(tmp_path):
    text = "max_num_seqs sets the maximum number of sequences"
    chunk = Chunck(
        file_path="data/raw/f0.md",
        first_character_index=0,
        last_character_index=len(text),
        size=len(text),
        text=text,
        title="t",
    )
    index = BM25Index()
    index.build([chunk])
    path = tmp_path / "index.json"
    index.save(path)
    return path


def assert_clean_failure(result, message: str, code: int = 1) -> None:
    assert result.returncode == code
    assert "Traceback" not in result.stderr
    last_line = result.stderr.strip().splitlines()[-1]
    assert last_line.startswith("Error:")
    assert message in last_line


class TestSuccess:
    def test_search(self, index_path):
        result = run_cli(
            "search",
            "max_num_seqs",
            "--k",
            "3",
            "--index_path",
            str(index_path),
        )
        assert result.returncode == 0
        assert json.loads(result.stdout)[0]["file_path"] == "data/raw/f0.md"

    def test_search_dataset_then_evaluate(self, tmp_path, index_path):
        dataset = tmp_path / "dataset.json"
        dataset.write_text(
            RagDataset(
                rag_questions=[
                    UnansweredQuestion(
                        question_id="q1", question="max_num_seqs"
                    ),
                ]
            ).model_dump_json()
        )
        out = tmp_path / "out"
        result = run_cli(
            "search_dataset",
            "--dataset_path",
            str(dataset),
            "--k",
            "5",
            "--save_directory",
            str(out),
            "--index_path",
            str(index_path),
        )
        assert result.returncode == 0
        assert "Saved student_search_results to" in result.stdout
        assert (out / "dataset.json").is_file()

    def test_commands_required_by_the_subject_exist(self):
        for command in ("index", "search", "search_dataset", "evaluate"):
            result = run_cli(command, "--help")
            assert "Traceback" not in result.stderr, command


class TestFailures:
    def test_empty_query(self, index_path):
        result = run_cli("search", "", "--index_path", str(index_path))
        assert_clean_failure(result, "empty")

    @pytest.mark.parametrize("k", ["0", "-1", "abc"])
    def test_invalid_k(self, index_path, k):
        result = run_cli(
            "search", "lora", "--k", k, "--index_path", str(index_path)
        )
        assert_clean_failure(result, "positive integer")

    def test_missing_index(self, tmp_path):
        result = run_cli(
            "search", "lora", "--index_path", str(tmp_path / "no.json")
        )
        assert_clean_failure(result, "not found")

    def test_corrupted_index(self, tmp_path):
        bad = tmp_path / "index.json"
        bad.write_text("{not json")
        result = run_cli("search", "lora", "--index_path", str(bad))
        assert_clean_failure(result, "not valid JSON")

    def test_missing_dataset(self, tmp_path, index_path):
        result = run_cli(
            "search_dataset",
            "--dataset_path",
            str(tmp_path / "no.json"),
            "--k",
            "3",
            "--save_directory",
            str(tmp_path),
            "--index_path",
            str(index_path),
        )
        assert_clean_failure(result, "was not found")

    def test_malformed_dataset(self, tmp_path, index_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        result = run_cli(
            "search_dataset",
            "--dataset_path",
            str(bad),
            "--k",
            "3",
            "--save_directory",
            str(tmp_path / "o"),
            "--index_path",
            str(index_path),
        )
        assert_clean_failure(result, "wrong format")

    def test_evaluate_missing_files(self, tmp_path):
        result = run_cli(
            "evaluate",
            "--student_search_results_path",
            str(tmp_path / "no.json"),
            "--dataset_path",
            str(tmp_path / "no2.json"),
        )
        assert_clean_failure(result, "search results file was not found")

    def test_index_missing_corpus(self, tmp_path):
        result = run_cli(
            "index",
            "--raw_dir",
            str(tmp_path / "nope"),
            "--index_path",
            str(tmp_path / "i.json"),
        )
        assert_clean_failure(result, "Corpus folder not found")

    def test_index_empty_corpus(self, tmp_path):
        (tmp_path / "raw").mkdir()
        result = run_cli(
            "index",
            "--raw_dir",
            str(tmp_path / "raw"),
            "--index_path",
            str(tmp_path / "i.json"),
        )
        assert_clean_failure(result, "Nothing to index")

    def test_wrong_dash_is_recognised(self, tmp_path):
        result = run_cli("index", "–-max_chunk_size", "1500")
        assert_clean_failure(result, "wrong dash")

    def test_index_invalid_chunk_size(self, tmp_path):
        (tmp_path / "raw").mkdir()
        result = run_cli(
            "index",
            "--raw_dir",
            str(tmp_path / "raw"),
            "--max_chunk_size",
            "0",
        )
        assert_clean_failure(result, "at least 1")

    def test_unknown_command_has_no_traceback(self):
        result = run_cli("nope")
        assert "Traceback" not in result.stderr
