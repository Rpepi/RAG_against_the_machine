from typing import Any

from src.utils.find_project_root import find_project_root


PROJECT_ROOT = find_project_root()
RAW_PREFIX = "data/raw"
DEFAULT_MAX_CHUNK_SIZE = 2000
MAX_CHUNK_SIZE = DEFAULT_MAX_CHUNK_SIZE
MIN_CHUNK_SIZE = 300
MODEL_NAME = "Qwen/Qwen3-0.6B"
# Any: the values mix int, bool and float, which `generate` accepts.
GENERATION_SETTINGS: dict[str, Any] = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}
