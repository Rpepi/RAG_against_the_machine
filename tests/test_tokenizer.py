from src.indexing.retrieval.tokenizer import split_identifier, tokenize


class TestSplitIdentifier:
    def test_snake_case(self):
        assert split_identifier("max_num_seqs") == ["max", "num", "seqs"]

    def test_camel_case(self):
        assert split_identifier("getModelConfig") == ["get", "model", "config"]

    def test_pascal_case_with_acronym(self):
        assert split_identifier("HTTPServerError") == ["http", "server", "error"]

    def test_digits_are_separate_parts(self):
        assert split_identifier("Qwen3") == ["qwen", "3"]

    def test_leading_and_double_underscores(self):
        assert split_identifier("__init__") == ["init"]
        assert split_identifier("a__b") == ["a", "b"]


class TestTokenize:
    def test_identifier_gives_whole_and_parts(self):
        tokens = tokenize("def getModelConfig(self): max_num_seqs")
        for expected in ("getmodelconfig", "get", "model", "config",
                         "max_num_seqs", "max", "num", "seqs", "def", "self"):
            assert expected in tokens

    def test_single_word_is_not_duplicated(self):
        assert tokenize("model") == ["model"]

    def test_question_matches_code_tokens(self):
        code = set(tokenize("self.max_num_seqs = max_num_seqs"))
        assert {"max_num_seqs", "seqs"} <= code & set(tokenize("What is max_num_seqs?"))
        assert "seqs" in code & set(tokenize("max number of seqs"))

    def test_is_lowercase(self):
        assert tokenize("MODEL") == ["model"]

    def test_mixed_case_acronym_keeps_the_whole_word(self):
        # camelCase splitting cannot tell LoRA from getHTTP: `lo` and `ra` are
        # noise, but the whole `lora` is there and questions are split the same way.
        assert "lora" in tokenize("LoRA")
        assert set(tokenize("LoRA")) == set(tokenize("lora adapter LoRA")) - {"adapter"}

    def test_duplicates_are_kept_for_term_frequency(self):
        assert tokenize("model model") == ["model", "model"]

    def test_one_character_tokens_are_dropped(self):
        assert tokenize("a + b = x") == []

    def test_markdown_and_punctuation_are_ignored(self):
        assert tokenize("## `--max-num-seqs` (int)") == ["max", "num", "seqs", "int"]

    def test_empty_and_blank(self):
        assert tokenize("") == []
        assert tokenize("  \n\t ") == []
        assert tokenize("___") == []
