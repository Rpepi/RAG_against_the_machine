import json
from pathlib import Path
import pathlib


def find_extension():
    extensions = set()
    for path in pathlib.Path(".").rglob("*"):
        if path.is_file():
            extensions.add(path.suffix)
    return extensions

def find_extension_dataset():
    extensions_docs = dict()
    extensions_code = dict()
    dir_extension = set()
    with open("data/datasets/AnsweredQuestions/dataset_code_public.json") as file:
        data = json.load(file)
        for item in data["rag_questions"]:
            for source in item["sources"]:
                file_path = source["file_path"]
                relative_path = Path(file_path).relative_to("data/raw/vllm-0.10.1/vllm/")
                dir = relative_path.parts[0]
                dir_extension.add(dir)
                extension = Path(file_path).suffix
                if  extension in extensions_code: 
                    n = extensions_code.get(extension)
                    extensions_code[extension] = n + 1
                else:
                    extensions_code[extension] =  1
    with open("data/datasets/AnsweredQuestions/dataset_docs_public.json") as file:
            data = json.load(file)
            for item in data["rag_questions"]:
                for source in item["sources"]:
                    file_path = source["file_path"]
                    extension = Path(file_path).suffix
                    if  extension in extensions_docs: 
                        n = extensions_docs.get(extension)
                        extensions_docs[extension] = n + 1
                    else:
                        extensions_docs[extension] =  1
    return extensions_docs, extensions_code, dir_extension


def find_median_span():
    spans = list()
    with open("data/datasets/AnsweredQuestions/dataset_code_public.json") as file:
            data = json.load(file)
            for item in data["rag_questions"]:
                for source in item["sources"]:
                    first_char = source["first_character_index"]
                    last_char = source["last_character_index"]
                    span = last_char - first_char
                    spans.append(span)
    spans.sort()
    size = len(spans)
    mid = size // 2
    return spans[0], spans[mid], spans[-1] ,spans










if __name__ == "__main__":
    min, median, max, spans = find_median_span()
    #print(f"Extensions docs & code & dir : [{find_extension_dataset()}]")
    print(f"min span: {min}")
    print(f"median span: {median}")
    print(f"max span: {max}")
    print(f"spans list: {spans}")