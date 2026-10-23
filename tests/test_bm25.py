import json
import math

import pytest
from src.retrieval.bm25 import BM25Index, IndexLoadError
from src.data_models.models import Chunck
from src.retrieval.tokenizer import tokenize

TEXTS = [
    "max_num_seqs sets the maximum number of sequences",
    "model model model model config",
    "the lora adapter",
]


def make_chunk(i: int, text: str) -> Chunck:
    return Chunck(
        file_path=f"data/raw/f{i}.md",
        first_character_index=10 * i,
        last_character_index=10 * i + len(text),
        size=len(text),
        text=text,
        title="t",
    )


@pytest.fixture
def index() -> BM25Index:
    bm25 = BM25Index()
    bm25.build([make_chunk(i, text) for i, text in enumerate(TEXTS)])
    return bm25


def hand_score(query: str, doc: int, k1: float = 1.5, b: float = 0.75):
    """BM25 written straight from the formula, independent from the class."""
    tokens = [tokenize(text) for text in TEXTS]
    n_chunks = len(tokens)
    avgdl = sum(len(t) for t in tokens) / n_chunks
    score = 0.0
    for term in set(tokenize(query)):
        tf = tokens[doc].count(term)
        if tf == 0:
            continue
        n_t = sum(term in t for t in tokens)
        idf = math.log(1 + (n_chunks - n_t + 0.5) / (n_t + 0.5))
        norm = 1 - b + b * len(tokens[doc]) / avgdl
        score += idf * tf * (k1 + 1) / (tf + k1 * norm)
    return score


class TestBuild:
    def test_state_after_build(self, index):
        assert index.doc_length == [10, 5, 3]
        assert index.avgdl == 6.0
        assert index.inverted_index["seqs"] == [[0, 1]]
        assert index.inverted_index["model"] == [[1, 4]]

    def test_sources_keep_the_chunk_position_and_range(self, index):
        source = index.sources[1]
        assert source.file_path == "data/raw/f1.md"
        assert source.first_character_index == 10
        assert source.last_character_index == 10 + len(TEXTS[1])

    def test_no_chunk_is_refused(self):
        with pytest.raises(ValueError, match="no chunk"):
            BM25Index().build([])

    def test_chunk_without_token_is_kept(self):
        bm25 = BM25Index()
        bm25.build([make_chunk(0, "?!"), make_chunk(1, "alpha beta")])
        assert bm25.doc_length == [0, 2]
        assert bm25.search("alpha", 5)[0].file_path == "data/raw/f1.md"

    def test_building_twice_gives_the_same_index(self):
        chunks = [make_chunk(0, "alpha beta"), make_chunk(1, "beta gamma")]
        once = BM25Index()
        once.build(chunks)
        twice = BM25Index()
        twice.build(chunks)
        twice.build(chunks)
        assert twice.inverted_index == once.inverted_index

    def test_k1_and_b_survive_rebuild(self):
        chunks = [make_chunk(0, "alpha beta"), make_chunk(1, "beta gamma")]
        bm25 = BM25Index(k1=2, b=0.5)
        bm25.build(chunks)
        bm25.build(chunks)
        assert bm25.k1 == 2
        assert bm25.b == 0.5


class TestSearch:
    def test_score_matches_the_formula(self, index):
        query = "what is max_num_seqs?"
        result = index.search(query, 3)
        assert len(result) == 1
        assert result[0].file_path == "data/raw/f0.md"
        assert result[0].score == pytest.approx(hand_score(query, 0))

    def test_ranking_is_by_decreasing_score(self, index):
        result = index.search("model config the", 3)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)
        for r in result:
            doc = int(
                r.file_path.removeprefix("data/raw/f").removesuffix(".md")
            )
            expected = hand_score("model config the", doc)
            assert r.score == pytest.approx(expected)

    def test_rare_word_beats_a_repeated_word_not_in_the_query(self, index):
        result = index.search("what is max_num_seqs?", 3)
        assert result[0].file_path == "data/raw/f0.md"

    def test_result_has_the_source_range(self, index):
        result = index.search("lora", 1)[0]
        assert result.first_character_index == 20
        assert result.last_character_index == 20 + len(TEXTS[2])

    def test_repeated_query_word_counts_once(self, index):
        once = index.search("lora", 1)[0].score
        assert index.search("lora lora lora", 1)[0].score == once

    def test_k_limits_the_number_of_results(self, index):
        assert len(index.search("the model lora", 1)) == 1

    def test_k_larger_than_the_matches(self, index):
        assert len(index.search("model lora", 50)) == 2

    @pytest.mark.parametrize("query", ["", "   ", "?!#", "zzzxyz"])
    def test_query_without_match_returns_nothing(self, index, query):
        assert index.search(query, 5) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_k_zero_or_negative_returns_nothing(self, index, k):
        assert index.search("lora", k) == []

    def test_query_with_one_unknown_word_still_matches_the_others(self, index):
        result = index.search("zzzxyz lora", 5)
        assert [r.file_path for r in result] == ["data/raw/f2.md"]


class TestSaveLoad:
    def test_round_trip_gives_the_same_results(self, index, tmp_path):
        path = tmp_path / "index.json"
        index.save(path)
        loaded = BM25Index.load(path)
        queries = ["what is max_num_seqs?", "model config", "lora", "zzz", ""]
        for query in queries:
            assert loaded.search(query, 5) == index.search(query, 5)

    def test_round_trip_keeps_the_settings(self, tmp_path):
        bm25 = BM25Index(k1=2.0, b=0.5)
        bm25.build([make_chunk(0, "alpha beta"), make_chunk(1, "beta")])
        path = tmp_path / "index.json"
        bm25.save(path)
        loaded = BM25Index.load(path)
        assert (loaded.k1, loaded.b, loaded.avgdl) == (2.0, 0.5, bm25.avgdl)
        assert loaded.doc_length == bm25.doc_length
        assert loaded.sources == bm25.sources

    def test_save_creates_missing_folders(self, index, tmp_path):
        path = tmp_path / "data" / "processed" / "index.json"
        index.save(path)
        assert path.is_file()

    def test_save_writes_to_the_given_path_only(self, index, tmp_path):
        path = tmp_path / "index.json"
        index.save(path)
        assert [p.name for p in tmp_path.iterdir()] == ["index.json"]

    def test_saved_file_has_the_expected_fields(self, index, tmp_path):
        path = tmp_path / "index.json"
        index.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert set(data) == {
            "k1",
            "b",
            "avgdl",
            "doc_length",
            "inverted_index",
            "sources",
        }


class TestLoadErrors:
    def load_text(self, tmp_path, content):
        path = tmp_path / "index.json"
        path.write_text(content, encoding="utf-8")
        return BM25Index.load(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(IndexLoadError, match="not found"):
            BM25Index.load(tmp_path / "nope.json")

    def test_directory_instead_of_file(self, tmp_path):
        with pytest.raises(IndexLoadError, match="Cannot read"):
            BM25Index.load(tmp_path)

    def test_invalid_json(self, tmp_path):
        with pytest.raises(IndexLoadError, match="not valid JSON"):
            self.load_text(tmp_path, "{not json")

    def test_wrong_encoding(self, tmp_path):
        path = tmp_path / "index.json"
        path.write_bytes(b'{"k1": "\xe9"}')
        with pytest.raises(IndexLoadError, match="not valid JSON"):
            BM25Index.load(path)

    def test_missing_field_is_named(self, tmp_path):
        with pytest.raises(IndexLoadError, match="no field 'k1'"):
            self.load_text(tmp_path, "{}")

    def test_json_that_is_not_an_object(self, tmp_path):
        with pytest.raises(IndexLoadError, match="malformed"):
            self.load_text(tmp_path, "[1, 2]")

    def test_malformed_source(self, tmp_path):
        content = json.dumps(
            {
                "k1": 1,
                "b": 1,
                "avgdl": 1,
                "doc_length": [],
                "inverted_index": {},
                "sources": [{"x": 1}],
            }
        )
        with pytest.raises(IndexLoadError, match="malformed"):
            self.load_text(tmp_path, content)

    def test_lengths_and_sources_must_match(self, tmp_path):
        content = json.dumps(
            {
                "k1": 1,
                "b": 1,
                "avgdl": 1,
                "doc_length": [1, 2, 3],
                "inverted_index": {},
                "sources": [],
            }
        )
        with pytest.raises(IndexLoadError, match="inconsistent"):
            self.load_text(tmp_path, content)

    def test_error_keeps_the_original_cause(self, tmp_path):
        with pytest.raises(IndexLoadError) as error:
            BM25Index.load(tmp_path / "nope.json")
        assert isinstance(error.value.__cause__, FileNotFoundError)
