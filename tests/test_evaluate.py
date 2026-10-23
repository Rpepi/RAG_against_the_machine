import json

import pytest
from src.data_models.models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion,
)
from src.evaluation.evaluate import (
    EvaluationError,
    count_without_sources,
    evaluate,
    is_found,
    load_model,
    overlap_ratio,
    question_recall,
    recall_at_k,
)


def source(path: str, first: int, last: int) -> MinimalSource:
    return MinimalSource(
        file_path=path, first_character_index=first, last_character_index=last
    )


def answered(question_id: str, sources: list[MinimalSource]):
    return AnsweredQuestion(
        question_id=question_id, question="?", sources=sources, answer="a"
    )


def found_by(question_id: str, sources: list[MinimalSource]):
    return MinimalSearchResults(
        question_id=question_id, question="?", retrieved_sources=sources
    )


EXPECTED = source("x", 4000, 4300)
COVERING = source("x", 3500, 5300)  # IoU = 300 / 1800


class TestOverlapRatio:
    def test_identical(self):
        assert overlap_ratio(source("x", 0, 100), source("x", 0, 100)) == 1.0

    def test_disjoint_ranges_give_zero_not_negative(self):
        assert overlap_ratio(source("x", 0, 100), source("x", 500, 600)) == 0.0

    def test_other_file_gives_zero_even_with_same_range(self):
        assert overlap_ratio(source("x", 0, 100), source("y", 0, 100)) == 0.0

    def test_partial_overlap(self):
        ratio = overlap_ratio(COVERING, EXPECTED)
        assert ratio == pytest.approx(300 / 1800)

    def test_is_symmetric(self):
        assert overlap_ratio(COVERING, EXPECTED) == overlap_ratio(
            EXPECTED, COVERING
        )

    def test_threshold_is_included(self):
        # 100 characters inside a 2000 characters chunk: exactly 0.05
        big, small = source("x", 0, 2000), source("x", 500, 600)
        assert overlap_ratio(big, small) == pytest.approx(0.05)
        assert is_found(small, [big])

    def test_just_under_the_threshold_is_not_found(self):
        big, small = source("x", 0, 2000), source("x", 500, 599)
        assert overlap_ratio(big, small) == pytest.approx(99 / 2000)
        assert not is_found(small, [big])

    def test_empty_ranges_do_not_divide_by_zero(self):
        assert overlap_ratio(source("x", 5, 5), source("x", 5, 5)) == 0.0


class TestIsFound:
    def test_no_result(self):
        assert is_found(EXPECTED, []) is False

    def test_any_result_is_enough(self):
        results = [source("y", 0, 9), source("x", 0, 10), COVERING]
        assert is_found(EXPECTED, results) is True

    def test_right_file_wrong_place(self):
        assert is_found(EXPECTED, [source("x", 0, 100)]) is False


class TestQuestionRecall:
    results = [source("a", 0, 9)] * 3 + [COVERING] + [source("b", 0, 9)]

    def test_cutoff_decides_what_is_looked_at(self):
        assert question_recall([EXPECTED], self.results, 3) == 0.0
        assert question_recall([EXPECTED], self.results, 4) == 1.0
        assert question_recall([EXPECTED], self.results, 10) == 1.0

    def test_share_of_expected_sources_found(self):
        expected = [EXPECTED, source("z", 0, 100)]
        assert question_recall(expected, self.results, 5) == 0.5

    def test_several_results_on_one_source_count_once(self):
        results = [COVERING, source("x", 3900, 4400)]
        assert question_recall([EXPECTED], results, 5) == 1.0

    def test_no_expected_source_is_undefined_not_zero(self):
        assert question_recall([], self.results, 5) is None

    def test_cutoff_zero(self):
        assert question_recall([EXPECTED], self.results, 0) == 0.0


class TestRecallAtK:
    truth = RagDataset(
        rag_questions=[
            answered("1", [EXPECTED]),
            answered("2", [source("z", 0, 100)]),
        ]
    )

    def results(self, *entries):
        return StudentSearchResults(search_results=list(entries), k=5)

    def test_everything_found(self):
        results = self.results(
            found_by("1", [COVERING]), found_by("2", [source("z", 0, 100)])
        )
        assert recall_at_k(results, self.truth, 5) == 1.0

    def test_question_missing_from_the_results_counts_zero(self):
        results = self.results(found_by("1", [COVERING]))
        assert recall_at_k(results, self.truth, 5) == 0.5

    def test_order_of_the_results_does_not_matter(self):
        results = self.results(
            found_by("2", [source("z", 0, 100)]), found_by("1", [COVERING])
        )
        assert recall_at_k(results, self.truth, 5) == 1.0

    def test_unknown_question_id_is_ignored(self):
        results = self.results(
            found_by("1", [COVERING]),
            found_by("2", [source("z", 0, 100)]),
            found_by("999", [source("x", 0, 9)]),
        )
        assert recall_at_k(results, self.truth, 5) == 1.0

    def test_recall_grows_with_the_cutoff(self):
        late = [source("a", 0, 9)] * 3 + [COVERING]
        results = self.results(found_by("1", late))
        truth = RagDataset(rag_questions=[answered("1", [EXPECTED])])
        assert recall_at_k(results, truth, 3) == 0.0
        assert recall_at_k(results, truth, 5) == 1.0

    def test_question_without_expected_source_is_left_out_of_the_mean(self):
        truth = RagDataset(
            rag_questions=[
                answered("1", [EXPECTED]),
                answered("2", []),
            ]
        )
        results = self.results(found_by("1", [COVERING]))
        assert recall_at_k(results, truth, 5) == 1.0

    def test_nothing_to_score(self):
        truth = RagDataset(rag_questions=[answered("1", [])])
        with pytest.raises(EvaluationError, match="no expected sources"):
            recall_at_k(self.results(), truth, 5)

    def test_dataset_without_answers(self):
        truth = RagDataset(
            rag_questions=[UnansweredQuestion(question_id="1", question="?")]
        )
        with pytest.raises(EvaluationError, match="AnsweredQuestions"):
            recall_at_k(self.results(), truth, 5)


def test_count_without_sources():
    truth = RagDataset(
        rag_questions=[
            answered("1", [EXPECTED]),
            answered("2", []),
            answered("3", []),
        ]
    )
    assert count_without_sources(truth) == 2


class TestLoadModel:
    def test_reads_a_valid_file(self, tmp_path):
        path = tmp_path / "res.json"
        path.write_text(
            StudentSearchResults(search_results=[], k=3).model_dump_json()
        )
        assert load_model(path, StudentSearchResults, "results").k == 3

    def test_missing_file(self, tmp_path):
        with pytest.raises(EvaluationError, match="not found") as error:
            load_model(tmp_path / "no.json", StudentSearchResults, "results")
        assert isinstance(error.value.__cause__, FileNotFoundError)

    def test_error_names_the_kind_of_file(self, tmp_path):
        with pytest.raises(EvaluationError, match="The dataset file"):
            load_model(tmp_path / "no.json", RagDataset, "dataset")

    def test_wrong_extension(self, tmp_path):
        path = tmp_path / "res.txt"
        path.write_text("{}")
        with pytest.raises(EvaluationError, match=r"must be a \.json"):
            load_model(path, StudentSearchResults, "results")

    def test_directory(self, tmp_path):
        with pytest.raises(EvaluationError):
            load_model(tmp_path, StudentSearchResults, "results")

    @pytest.mark.parametrize("content", ["{not json", "{}", "[1, 2]", ""])
    def test_invalid_content(self, tmp_path, content):
        path = tmp_path / "res.json"
        path.write_text(content)
        with pytest.raises(EvaluationError, match="wrong format") as error:
            load_model(path, StudentSearchResults, "results")
        assert error.value.__cause__ is not None


class TestEvaluate:
    @pytest.fixture
    def files(self, tmp_path):
        truth = RagDataset(
            rag_questions=[
                answered("1", [EXPECTED]),
                answered("2", [source("z", 0, 100)]),
            ]
        )
        dataset = tmp_path / "truth.json"
        dataset.write_text(truth.model_dump_json())
        return tmp_path, dataset

    def write_results(self, folder, *entries):
        path = folder / "results.json"
        path.write_text(
            StudentSearchResults(
                search_results=list(entries), k=10
            ).model_dump_json()
        )
        return path

    def test_perfect_results_give_one_everywhere(self, files, capsys):
        folder, dataset = files
        results = self.write_results(
            folder,
            found_by("1", [EXPECTED]),
            found_by("2", [source("z", 0, 100)]),
        )
        evaluate(results, dataset)
        out = capsys.readouterr().out
        for cutoff in (1, 3, 5, 10):
            assert f"Recall@{cutoff}: 1.000" in out

    def test_wrong_paths_give_zero(self, files, capsys):
        folder, dataset = files
        results = self.write_results(
            folder,
            found_by("1", [source("other", 4000, 4300)]),
            found_by("2", [source("other", 0, 100)]),
        )
        evaluate(results, dataset)
        assert "Recall@5: 0.000" in capsys.readouterr().out

    def test_warns_about_questions_without_expected_sources(
        self, tmp_path, capsys
    ):
        truth = RagDataset(
            rag_questions=[
                answered("1", [EXPECTED]),
                answered("2", []),
            ]
        )
        dataset = tmp_path / "truth.json"
        dataset.write_text(truth.model_dump_json())
        results = self.write_results(tmp_path, found_by("1", [EXPECTED]))
        evaluate(results, dataset)
        captured = capsys.readouterr()
        assert "1 question(s) without expected sources" in captured.err
        assert "Recall@5: 1.000" in captured.out

    def test_unanswered_dataset_is_refused(self, tmp_path):
        dataset = tmp_path / "truth.json"
        dataset.write_text(
            json.dumps(
                {"rag_questions": [{"question_id": "1", "question": "?"}]}
            )
        )
        results = self.write_results(tmp_path, found_by("1", [EXPECTED]))
        with pytest.raises(EvaluationError, match="AnsweredQuestions"):
            evaluate(results, dataset)

    def test_missing_results_file(self, files):
        folder, dataset = files
        with pytest.raises(EvaluationError, match="search results"):
            evaluate(folder / "nope.json", dataset)
