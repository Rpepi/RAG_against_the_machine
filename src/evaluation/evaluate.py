from src.data_models.models import (
    MinimalSource,
    StudentSearchResults,
    RagDataset,
    AnsweredQuestion,
)
from pathlib import Path
import sys
from src.utils.load_model import EvaluationError, load_model

FOUND_IOU = 0.05


def overlap_ratio(retrieved: MinimalSource, answer: MinimalSource) -> float:
    """Intersection over union of the character ranges of two sources.

    Returns:
        The IoU, or 0.0 when the files differ or a range is empty.
    """
    if retrieved.file_path != answer.file_path:
        return 0.0

    a = retrieved.first_character_index
    b = retrieved.last_character_index
    c = answer.first_character_index
    d = answer.last_character_index

    if a == b or c == d:
        return 0.0

    intersection = max(0, min(b, d) - max(a, c))
    union = (b - a) + (d - c) - intersection
    if union == 0:
        return 0.0
    IoU = intersection / union

    return IoU


def is_found(expected: MinimalSource, retrieved: list[MinimalSource]) -> bool:
    """Whether one of `retrieved` overlaps `expected` by FOUND_IOU or more."""
    result = any(
        overlap_ratio(source, expected) >= FOUND_IOU for source in retrieved
    )
    return result


def question_recall(
    expected_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int,
) -> float | None:
    """Share of the expected sources found in the first `k` retrieved ones.

    Returns:
        The recall of the question, or None when it has no expected source
        (the recall is then undefined).
    """
    top = retrieved_sources[:k]
    result = 0
    for source in expected_sources:
        found = is_found(source, top)
        if found:
            result += 1
    if not expected_sources:
        return None
    response = result / len(expected_sources)
    return response


def recall_at_k(
    result: StudentSearchResults, truth: RagDataset, k: int
) -> float:
    """Mean recall@k over the answered questions of `truth`.

    A question missing from `result` counts as 0; questions without expected
    sources are left out of the mean.

    Raises:
        EvaluationError: no question of `truth` has expected sources.
    """
    scores: list[float] = []

    question_sources: dict[str, list[MinimalSource]] = {}
    search_result = result.search_results
    for search in search_result:
        question_sources[search.question_id] = search.retrieved_sources
    for question in truth.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            continue
        sources = question_sources.get(question.question_id, [])
        score = question_recall(question.sources, sources, k)
        if score is None:
            continue
        scores.append(score)
    if not scores:
        raise EvaluationError(
            "This dataset has no expected sources: "
            "use the AnsweredQuestions file"
        )
    value = sum(scores) / len(scores)
    return value


def count_without_sources(truth: RagDataset) -> int:
    """Number of answered questions that have no expected source."""
    return sum(
        1
        for question in truth.rag_questions
        if isinstance(question, AnsweredQuestion) and not question.sources
    )


def evaluate(
    student_search_results_path: Path | str, dataset_path: Path | str
) -> None:
    """Print recall@1, 3, 5 and 10 of a search results file.

    Args:
        student_search_results_path: file written by `search_dataset`.
        dataset_path: the AnsweredQuestions dataset (expected sources).
    """
    results = load_model(
        Path(student_search_results_path),
        StudentSearchResults,
        "search results",
    )
    truth = load_model(Path(dataset_path), RagDataset, "dataset")
    skipped = count_without_sources(truth)
    if skipped:
        print(
            f"Warning: {skipped} question(s) without expected sources "
            "were ignored",
            file=sys.stderr,
        )
    print(
        " ".join(
            f"Recall@{cutoff}: {recall_at_k(results, truth, cutoff):.3f}"
            for cutoff in (1, 3, 5, 10)
        )
    )
