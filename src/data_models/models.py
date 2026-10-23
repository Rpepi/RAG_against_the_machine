from pydantic import BaseModel, Field
import uuid


class MinimalSource(BaseModel):
    """A location in the corpus: a file and a range of characters.

    `last_character_index` is exclusive, as in `text[first:last]`.
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class ScoredSource(MinimalSource):
    """A source with its BM25 score, as returned by `BM25Index.search`."""
    score: float


class Chunck(MinimalSource):
    """A piece of a file: its location, size, text and title.

    The title is the section or the definition the chunk belongs to.
    """
    size: int
    text: str
    title: str


class UnansweredQuestion(BaseModel):
    """A question of a dataset, with an id (a random UUID by default)."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its expected sources and answer: the ground truth."""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset file: a list of questions, answered or not."""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question, best first."""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question and the generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Output of `search_dataset`: the results of each question and `k`."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of `answer_dataset`: the results with answers, and `k`."""
    search_results: list[MinimalAnswer]
    k: int
