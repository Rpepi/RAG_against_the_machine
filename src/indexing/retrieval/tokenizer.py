import re

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
MIN_TOKEN_LENGTH = 2


def split_identifier(word: str) -> list[str]:
    """Lowercase parts of an identifier: `getModelConfig` -> get, model, config.

    Splits on underscores, on camelCase boundaries and between letters and digits.
    """
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(part.lower() for part in PART_RE.findall(piece))
    return parts


def tokenize(text: str) -> list[str]:
    """Tokens of a chunk or of a question, the same way for both.

    Each identifier gives its whole lowercase form, so a question quoting
    `max_num_seqs` matches exactly, then its parts when there are several, so
    a paraphrase like "max number of seqs" matches too. Duplicates are kept:
    BM25 needs the term frequencies. One-character tokxens are dropped.
    """
    tokens: list[str] = []
    for word in WORD_RE.findall(text):
        whole = word.strip("_").lower()
        parts = split_identifier(word)
        candidates = [whole] + (parts if len(parts) > 1 else [])
        tokens.extend(t for t in candidates if len(t) >= MIN_TOKEN_LENGTH)
    return tokens
