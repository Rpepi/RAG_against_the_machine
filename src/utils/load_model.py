from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from src.utils.errors import RagError

T = TypeVar("T", bound=BaseModel)


class EvaluationError(RagError):
    """A results file or a dataset cannot be read or evaluated."""


def load_model(path: Path | str, model: type[T], label: str) -> T:
    """Read a JSON file and validate it as `model`.

    Args:
        path: file to read.
        model: pydantic model the content must follow.
        label: what the file is ("dataset", "search results"), used in the
            error messages.

    Raises:
        EvaluationError: the file is missing, unreadable, not a `.json`
            file, or its content does not follow `model`.
    """
    path = Path(path)
    if path.suffix != ".json":
        raise EvaluationError(f"The {label} file must be a .json file.")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise EvaluationError(f"The {label} file was not found: {path}") from e
    except IsADirectoryError as e:
        raise EvaluationError(f"The {label} path is a directory") from e
    except UnicodeDecodeError as e:
        raise EvaluationError(
            f"The {label} file {path} is not valid UTF-8 text"
        ) from e
    except OSError as e:
        raise EvaluationError(
            f"Cannot read the {label} file {path}: {e.strerror}"
        ) from e
    try:
        result: T = model.model_validate_json(text)
        return result
    except ValidationError as e:
        raise EvaluationError(
            f"The {label} file {path} has a wrong format"
        ) from e
