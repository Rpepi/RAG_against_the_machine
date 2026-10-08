from pathlib import Path

def find_project_root(start: Path = Path(__file__)) -> Path:
    for parent in start.resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml introuvable")

