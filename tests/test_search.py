import json

import pytest
from src.data_models.models import (
    AnsweredQuestion,
    Chunck,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.evaluation.evaluate import recall_at_k
from src.retrieval.bm25 import BM25Index, IndexLoadError
from src.retrieval.search import check_k, search, search_dataset
from src.utils.errors import RagError
from src.utils.load_model import EvaluationError

TEXTS = [
    "max_num_seqs sets the maximum number of sequences",
    "model model model config",
    "the lora adapter is loaded with a lora request",
    "unrelated words about caching",
]


def make_chunk(i: int, text: str) -> Chunck:
    return Chunck(
        file_path=f"data/raw/f{i}.md",
        first_character_index=0,
        last_character_index=len(text),
        size=len(text),
        text=text,
        title="t",
    )


@pytest.fixture
def index_path(tmp_path):
    index = BM25Index()
    index.build([make_chunk(i, text) for i, text in enumerate(TEXTS)])
    path = tmp_path / "index.json"
    index.save(path)
    return path


def write_dataset(path, questions):
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = RagDataset(rag_questions=questions)
    path.write_text(dataset.model_dump_json(), encoding="utf-8")
    return path


QUESTIONS = [
    UnansweredQuestion(question_id="q1", question="what is max_num_seqs?"),
    UnansweredQuestion(question_id="q2", question="lora adapter request"),
    UnansweredQuestion(question_id="q3", question="zzzxyz"),
]


class TestCheckK:
    @pytest.mark.parametrize("k", [1, 5, 100])
    def test_valid(self, k):
        assert check_k(k) == k

    @pytest.mark.parametrize("k", [0, -1, "5", 2.5, None, True])
    def test_invalid(self, k):
        with pytest.raises(RagError, match="positive integer"):
            check_k(k)


class TestSearchDataset:
    def run(self, tmp_path, index_path, k=2, questions=QUESTIONS):
        dataset = write_dataset(
            tmp_path / "datasets" / "dataset_docs_public.json", questions
        )
        save_directory = tmp_path / "out" / "UnansweredQuestions"
        path = search_dataset(dataset, k, save_directory, index_path)
        return dataset, save_directory, path

    def load(self, path) -> StudentSearchResults:
        return StudentSearchResults.model_validate_json(path.read_text())

    def test_output_file_keeps_the_dataset_name(self, tmp_path, index_path):
        _, save_directory, path = self.run(tmp_path, index_path)
        assert path == save_directory / "dataset_docs_public.json"
        assert path.is_file()

    def test_output_follows_the_model(self, tmp_path, index_path):
        _, _, path = self.run(tmp_path, index_path, k=3)
        output = self.load(path)
        assert output.k == 3
        assert [r.question_id for r in output.search_results] == [
            "q1",
            "q2",
            "q3",
        ]

    def test_each_question_has_its_own_sources(self, tmp_path, index_path):
        # regression: one shared list made the sources pile up
        _, _, path = self.run(tmp_path, index_path)
        results = self.load(path).search_results
        assert results[0].retrieved_sources[0].file_path == "data/raw/f0.md"
        assert results[1].retrieved_sources[0].file_path == "data/raw/f2.md"
        assert results[2].retrieved_sources == []

    def test_at_most_k_sources_per_question(self, tmp_path, index_path):
        _, _, path = self.run(tmp_path, index_path, k=1)
        results = self.load(path).search_results
        assert all(len(r.retrieved_sources) <= 1 for r in results)

    def test_no_score_in_the_file(self, tmp_path, index_path):
        _, _, path = self.run(tmp_path, index_path)
        raw = json.loads(path.read_text())
        source = raw["search_results"][0]["retrieved_sources"][0]
        assert set(source) == {
            "file_path",
            "first_character_index",
            "last_character_index",
        }

    def test_sources_are_the_chunk_ranges(self, tmp_path, index_path):
        _, _, path = self.run(tmp_path, index_path)
        source = self.load(path).search_results[0].retrieved_sources[0]
        assert (source.first_character_index, source.last_character_index) == (
            0,
            len(TEXTS[0]),
        )

    def test_creates_the_missing_folders(self, tmp_path, index_path):
        _, save_directory, _ = self.run(tmp_path, index_path)
        assert save_directory.is_dir()

    def test_accepts_strings_like_fire_gives(self, tmp_path, index_path):
        dataset = write_dataset(tmp_path / "d.json", QUESTIONS)
        path = search_dataset(
            str(dataset), 2, str(tmp_path / "out"), str(index_path)
        )
        assert path.is_file()

    def test_search_then_evaluate(self, tmp_path, index_path):
        truth = RagDataset(
            rag_questions=[
                AnsweredQuestion(
                    question_id="q1",
                    question="what is max_num_seqs?",
                    sources=[
                        MinimalSource(
                            file_path="data/raw/f0.md",
                            first_character_index=0,
                            last_character_index=len(TEXTS[0]),
                        )
                    ],
                    answer="a",
                ),
                AnsweredQuestion(
                    question_id="q2",
                    question="lora adapter request",
                    sources=[
                        MinimalSource(
                            file_path="data/raw/f3.md",  # the wrong file
                            first_character_index=0,
                            last_character_index=10,
                        )
                    ],
                    answer="a",
                ),
            ]
        )
        dataset = write_dataset(tmp_path / "d.json", truth.rag_questions)
        path = search_dataset(dataset, 3, tmp_path / "out", index_path)
        results = self.load(path)
        assert recall_at_k(results, truth, 5) == 0.5

    def test_missing_index(self, tmp_path):
        dataset = write_dataset(tmp_path / "d.json", QUESTIONS)
        with pytest.raises(IndexLoadError, match="not found"):
            search_dataset(dataset, 2, tmp_path / "out", tmp_path / "no.json")

    def test_missing_dataset(self, tmp_path, index_path):
        with pytest.raises(EvaluationError, match="dataset file"):
            search_dataset(
                tmp_path / "no.json", 2, tmp_path / "out", index_path
            )

    def test_malformed_dataset(self, tmp_path, index_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(EvaluationError, match="wrong format"):
            search_dataset(bad, 2, tmp_path / "out", index_path)

    @pytest.mark.parametrize("k", [0, -3])
    def test_invalid_k(self, tmp_path, index_path, k):
        dataset = write_dataset(tmp_path / "d.json", QUESTIONS)
        with pytest.raises(RagError, match="positive integer"):
            search_dataset(dataset, k, tmp_path / "out", index_path)

    def test_never_overwrites_the_dataset(self, tmp_path, index_path):
        dataset = write_dataset(tmp_path / "d.json", QUESTIONS)
        with pytest.raises(RagError, match="overwrite"):
            search_dataset(dataset, 2, tmp_path, index_path)
        assert RagDataset.model_validate_json(dataset.read_text())

    def test_empty_dataset(self, tmp_path, index_path):
        dataset = write_dataset(tmp_path / "d.json", [])
        path = search_dataset(dataset, 2, tmp_path / "out", index_path)
        assert self.load(path).search_results == []


class TestSearch:
    def test_prints_a_json_list(self, index_path, capsys):
        search("what is max_num_seqs?", 2, index_path)
        printed = json.loads(capsys.readouterr().out)
        assert printed[0]["file_path"] == "data/raw/f0.md"
        assert set(printed[0]) == {
            "file_path",
            "first_character_index",
            "last_character_index",
        }

    def test_nothing_found_prints_an_empty_list(self, index_path, capsys):
        search("zzzxyz", 3, index_path)
        assert json.loads(capsys.readouterr().out) == []

    def test_k_limits_the_list(self, index_path, capsys):
        search("the model lora", 1, index_path)
        assert len(json.loads(capsys.readouterr().out)) == 1

    @pytest.mark.parametrize("query", ["", "   ", "\n"])
    def test_empty_query(self, index_path, query):
        with pytest.raises(RagError, match="empty"):
            search(query, 3, index_path)

    def test_query_given_as_a_number_by_fire(self, index_path, capsys):
        search(2024, 3, index_path)  # type: ignore[arg-type]
        assert json.loads(capsys.readouterr().out) == []

    def test_invalid_k(self, index_path):
        with pytest.raises(RagError, match="positive integer"):
            search("lora", 0, index_path)

    def test_missing_index(self, tmp_path):
        with pytest.raises(IndexLoadError):
            search("lora", 3, tmp_path / "no.json")
