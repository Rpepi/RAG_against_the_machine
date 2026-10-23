from src.data_models.models import (
    MinimalSource,
    Chunck,
    ScoredSource,
)
from src.config import PROJECT_ROOT
from src.utils.errors import RagError
from src.retrieval.tokenizer import tokenize
from collections import Counter
from pathlib import Path
from pydantic import ValidationError
import heapq
import json
import math


DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "bm25_index.json"


class IndexLoadError(RagError):
    """An index file cannot be read or is not a valid index."""


class BM25Index:
    """BM25 index over chunks: inverted index, chunk lengths and locations.

    `k1` sets how fast repeated terms saturate and `b` how much long chunks
    are penalised. Chunks are identified by their position in `sources`.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Create an empty index with the BM25 parameters `k1` and `b`."""
        self.k1 = k1  # saturation parameter
        self.b = b  # length penalty parameter
        self._reset()

    def _reset(self) -> None:
        """Empty the index; `k1` and `b` are kept."""
        # token -> [[chunk_id, tf], ...], e.g. {"docker": [[3, 12], [4, 8]]}
        self.inverted_index: dict[str, list[list[int]]] = {}
        self.doc_length: list[int] = []  # n of tokens per chunk
        self.avgdl: float = 1.0  # average of tokens per chunk
        self.sources: list[MinimalSource] = []

    def build(self, chunks: list[Chunck]) -> None:
        """Index `chunks`, replacing any previous content.

        Raises:
            ValueError: `chunks` is empty.
        """
        if not chunks:
            raise ValueError("no chunk to index")
        self._reset()  # reset the object
        for chunck_id, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)

            for token, tf in Counter(tokens).items():
                postings = self.inverted_index.setdefault(token, [])
                postings.append([chunck_id, tf])

            self.doc_length.append(len(tokens))
            self.sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
        self.avgdl = sum(self.doc_length) / len(self.doc_length)

    def save(self, path: Path) -> None:
        """Write the index as JSON to `path`, creating the missing folders."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, object] = {
            "k1": self.k1,
            "b": self.b,
            "inverted_index": self.inverted_index,
            "doc_length": self.doc_length,
            "avgdl": self.avgdl,
            "sources": [source.model_dump() for source in self.sources],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _idf(self, token: str) -> float:
        """Inverse document frequency of `token` (a variant that stays > 0)."""
        # n of chunks that contain the token
        n_t = len(self.inverted_index.get(token, []))
        N = len(self.doc_length)  # n total of chunks
        return math.log(1 + (N - n_t + 0.5) / (n_t + 0.5))

    def _bm25(self, idf: float, tf: int, chunk_id: int) -> float:
        """BM25 score in chunk `chunk_id` of a query token with this IDF."""
        length_ratio = self.doc_length[chunk_id] / self.avgdl
        norm = 1 - self.b + self.b * length_ratio
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)

    def search(self, query: str, k: int) -> list[ScoredSource]:
        """The `k` chunks with the best BM25 score for `query`, best first.

        Only the chunks that share a token with the query are scored.
        """
        scores: dict[int, float] = {}
        query_tokens = set(tokenize(query))

        for token in query_tokens:
            idf = self._idf(token)
            for chunk_id, tf in self.inverted_index.get(token, []):
                bm25 = self._bm25(idf, tf, chunk_id)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25

        top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
        results: list[ScoredSource] = []
        for chunk_id, score in top:
            source = self.sources[chunk_id]
            results.append(
                ScoredSource(
                    file_path=source.file_path,
                    first_character_index=source.first_character_index,
                    last_character_index=source.last_character_index,
                    score=score,
                )
            )
        return results

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        """Read an index saved by `save`.

        Raises:
            IndexLoadError: the file is missing, unreadable, not valid JSON,
                or does not describe a valid index.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            index = cls(k1=data["k1"], b=data["b"])
            index.avgdl = data["avgdl"]
            index.doc_length = data["doc_length"]
            index.inverted_index = data["inverted_index"]
            index.sources = [MinimalSource(**s) for s in data["sources"]]
        except FileNotFoundError as e:
            raise IndexLoadError(f"Index file not found: {path}") from e
        except OSError as e:
            raise IndexLoadError(
                f"Cannot read the index file {path}: {e.strerror}"
            ) from e
        except ValidationError as e:  # before ValueError, its parent class
            raise IndexLoadError(
                f"The index file {path} is malformed: {e}"
            ) from e
        except ValueError as e:  # invalid JSON or encoding
            raise IndexLoadError(
                f"The index file {path} is not valid JSON: {e}"
            ) from e
        except KeyError as e:
            raise IndexLoadError(
                f"The index file {path} has no field {e}"
            ) from e
        except TypeError as e:
            raise IndexLoadError(
                f"The index file {path} is malformed: {e}"
            ) from e
        if len(index.doc_length) != len(index.sources):
            raise IndexLoadError(
                f"The index file {path} is inconsistent: "
                f"{len(index.doc_length)} lengths for "
                f"{len(index.sources)} sources"
            )
        return index
