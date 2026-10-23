import functools
import sys
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import fire

from src.evaluation.evaluate import evaluate
from src.retrieval.indexer import indexer
from src.retrieval.search import search, search_dataset
from src.answer.answer import answer, answer_dataset
from src.utils.errors import RagError

P = ParamSpec("P")
R = TypeVar("R")


def fail(message: str, code: int) -> None:
    """Print an error on stderr and leave with a non-zero status."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def safe(command: Callable[P, R]) -> Callable[P, R | None]:
    """Turn the errors of a command into a message and an exit status.

    Expected problems (`RagError`, a bad value, an unreadable file) exit
    with status 1, anything else with status 2: the user never sees a
    traceback.
    """

    @functools.wraps(command)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R | None:
        """Run the command; on error, print a message and exit."""
        try:
            command(*args, **kwargs)
        except (RagError, ValueError, OSError) as e:
            fail(str(e), 1)
        except KeyboardInterrupt:
            fail("interrupted", 130)
        except Exception as e:
            fail(f"unexpected {type(e).__name__}: {e}", 2)
        return None

    return wrapper


COMMANDS: dict[str, Callable[..., object]] = {
    "index": indexer,
    "search": search,
    "search_dataset": search_dataset,
    "evaluate": evaluate,
    "answer": answer,
    "answer_dataset": answer_dataset
}


def main() -> None:
    """Run the command line: `python -m src <command> [options]`."""
    fire.Fire({name: safe(func) for name, func in COMMANDS.items()})
