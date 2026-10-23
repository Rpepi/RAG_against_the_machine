import pytest
from src.chunking import common


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A fake project root; returns `write(name, content) -> Path`.

    Files are written byte for byte (no newline translation), like
    `read_source` reads them.
    """
    root = tmp_path.resolve()
    monkeypatch.setattr(common, "PROJECT_ROOT", root)

    def write(name: str, content: str):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return path

    return write
