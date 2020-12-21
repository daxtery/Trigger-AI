from dataclasses import dataclass


@dataclass(frozen=True)
class Scoring:
    with_tag: str
    similarity_score: float
    
    is_match: bool