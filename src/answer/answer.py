from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import GENERATION_SETTINGS, MODEL_NAME, PROJECT_ROOT
from src.data_models.models import (
    MinimalAnswer,
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retrieval.search import retrieve
from src.utils.errors import RagError
from src.utils.load_model import load_model

PROMPT = (
    "Answer the question using only the context below.\n"
    "Answer in one to three plain sentences, without lists, tables, "
    "markdown or emojis.\n"
    "If the context does not contain the answer, say "
    "“I cannot answer from the provided documents.”\n"
)


def find_answer_sources(sources_info: list[MinimalSource]) -> list[str]:
    """Read the text of each source from its file.

    Files are opened with `newline=""`, like the chunker, so that the
    character offsets of the sources match the text exactly.

    Args:
        sources_info: the locations (file and character range) to read.

    Returns:
        The text of each source, in the same order.

    Raises:
        OSError: a file cannot be read.
    """
    answers: list[str] = []
    for source in sources_info:
        path = Path(PROJECT_ROOT / source.file_path)
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
            first = source.first_character_index
            last = source.last_character_index
            answers.append(text[first:last])
    return answers


def answer(query: str, k: int) -> None:
    """Answer one question from the passages that BM25 retrieves for it.

    The `k` best passages are given to Qwen3-0.6B as context, and the
    generated answer is printed.

    Args:
        query: the question.
        k: number of passages retrieved and put in the prompt.

    Raises:
        RagError: `k` is invalid, the query is empty or the index is
            missing.
    """
    sources_info: list[MinimalSource] = retrieve(query, k)
    sources_answer = find_answer_sources(sources_info)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Typed as Any: the type hints of transformers' `generate` cannot be
    # satisfied by a loaded model (even `Qwen3ForCausalLM` fails mypy).
    model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
    context = "\n\n".join(
        f"[{i}] {source.file_path}\n{passage}"
        for i, (source, passage) in enumerate(
            zip(sources_info, sources_answer), 1
        )
    )

    messages = [
            {
                "role": "system",
                "content": PROMPT
            },
            {
                "role": "user",
                "content": f"context:\n{context}\n\nQuestion: {query}"
            },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # return_tensors="pt" gives a PyTorch tensor instead of a python list
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        **GENERATION_SETTINGS,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = str(tokenizer.decode(output_ids, skip_special_tokens=True))
    print("content:", content.strip("\n"))


def answer_dataset(
    student_search_results_path: Path | str,
    save_directory: Path | str,
) -> None:
    """Answer every question of a search results file and save them.

    The input is the file written by `search_dataset`; the output follows
    `StudentSearchResultsAndAnswer`, has the same name and is written in
    `save_directory`. Relative paths start at the project root. The model is
    loaded once for all the questions.

    Args:
        student_search_results_path: a `StudentSearchResults` JSON file.
        save_directory: folder of the output file, created if missing.

    Raises:
        EvaluationError: the results file is missing or malformed.
        RagError: the output file cannot be written.
    """
    search_path = Path(PROJECT_ROOT / student_search_results_path)
    save_dir = Path(PROJECT_ROOT / save_directory)
    save_path = save_dir / search_path.name
    student_search_results: StudentSearchResults = load_model(
        search_path,
        StudentSearchResults,
        "search results",
    )
    student_search_result_answer = StudentSearchResultsAndAnswer(
        search_results=[], k=student_search_results.k
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Typed as Any: see the comment in `answer`.
    model: Any = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RagError(f"Cannot create {save_dir}: {e.strerror}") from e

    questions = student_search_results.search_results
    for question in tqdm(questions, desc="Answering"):
        sources_answer = find_answer_sources(question.retrieved_sources)

        context = "\n\n".join(
            f"[{i}] {source.file_path}\n{passage}"
            for i, (source, passage) in enumerate(
                zip(question.retrieved_sources, sources_answer), 1
            )
        )

        messages = [
                {
                    "role": "system",
                    "content": PROMPT
                },
                {
                    "role": "user",
                    "content": (
                        f"context:\n{context}\n\n"
                        f"Question: {question.question}"
                    ),
                },
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # return_tensors="pt" gives a PyTorch tensor instead of a python list
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            **GENERATION_SETTINGS,
        )
        prompt_length = len(model_inputs.input_ids[0])
        output_ids = generated_ids[0][prompt_length:].tolist()
        content = str(tokenizer.decode(output_ids, skip_special_tokens=True))
        minimal_answer = MinimalAnswer(
            answer=content.strip("\n"),
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=question.retrieved_sources,
        )
        student_search_result_answer.search_results.append(minimal_answer)
    try:
        save_path.write_text(
            student_search_result_answer.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError as e:
        raise RagError(f"Cannot write {save_path}: {e.strerror}") from e
    print(f"Saved student_search_result_and_answer to {save_path}")
