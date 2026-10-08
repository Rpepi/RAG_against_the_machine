from pydantic import BaseModel


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int


class Chunck(MinimalSource):
    size: int
    text: str
    title: str

class ScoredSource(MinimalSource):
    score:float