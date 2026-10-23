from pathlib import Path


def find_project_root(start: Path = Path(__file__)) -> Path:
    """Return the first parent of `start` that contains `pyproject.toml`.

    Raises:
        FileNotFoundError: no parent contains it.
    """
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml introuvable")
