import json
from pathlib import Path

from tqdm import tqdm

from src.data_models.models import (
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    ScoredSource,
    StudentSearchResults,
)
from src.retrieval.bm25 import DEFAULT_INDEX_PATH, BM25Index
from src.utils.errors import RagError
from src.utils.load_model import load_model


def check_k(k: object) -> int:
    """Return `k` if it is a positive integer, raise RagError otherwise."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise RagError(f"k must be a positive integer, got {k!r}")
    return k


def to_minimal(hit: ScoredSource) -> MinimalSource:
    """Drop the score: the output files only hold the source location."""
    return MinimalSource(
        file_path=hit.file_path,
        first_character_index=hit.first_character_index,
        last_character_index=hit.last_character_index,
    )


def search_minimal(
    index: BM25Index, query: str, k: int
) -> list[MinimalSource]:
    """The `k` best sources of `query`, best first, without their score."""
    return [to_minimal(hit) for hit in index.search(query, k)]


def retrieve(
    query: str, k: int = 10, index_path: Path | str = DEFAULT_INDEX_PATH
) -> list[MinimalSource]:
    """Étapes 1 à 4 : vérifier, charger l'index, chercher. N'affiche rien."""
    k = check_k(k)
    query = str(query)
    if not query.strip():
        raise RagError("The query is empty.")
    index = BM25Index.load(Path(index_path))
    return search_minimal(index, query, k)


def search(
    query: str, k: int = 10, index_path: Path | str = DEFAULT_INDEX_PATH
) -> None:
    """La commande : chercher, puis afficher."""
    sources = retrieve(query, k, index_path)
    print(json.dumps([s.model_dump() for s in sources], indent=2))


def search_dataset(
    dataset_path: Path | str,
    k: int,
    save_directory: Path | str,
    index_path: Path | str = DEFAULT_INDEX_PATH,
) -> Path:
    """Search every question of a dataset and write the results as JSON.

    The output file has the name of the dataset, inside `save_directory`,
    and follows the `StudentSearchResults` model.

    Args:
        dataset_path: JSON file with the questions.
        k: number of sources kept for each question.
        save_directory: folder of the output file, created if missing.
        index_path: index written by the `index` command.

    Returns:
        The path of the file written.

    Raises:
        RagError: `k` is invalid, a file is missing or malformed, or the
            index cannot be loaded.
    """
    k = check_k(k)
    dataset_path = Path(dataset_path)
    save_directory = Path(save_directory)
    dataset = load_model(dataset_path, RagDataset, "dataset")
    index = BM25Index.load(Path(index_path))
    results = [
        MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=search_minimal(index, question.question, k),
        )
        for question in tqdm(dataset.rag_questions, desc="Searching")
    ]
    output = StudentSearchResults(search_results=results, k=k)
    save_path = save_directory / dataset_path.name
    if save_path.resolve() == dataset_path.resolve():
        raise RagError("The output file would overwrite the dataset.")
    try:
        save_directory.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            output.model_dump_json(indent=2), encoding="utf-8"
        )
    except OSError as e:
        raise RagError(f"Cannot write {save_path}: {e.strerror}") from e
    print(f"Saved student_search_results to {save_path}")
    return save_path
