import numpy
from interference.metrics.match import similarity_metric
from typing import Any, Dict, Optional

from dataclasses import dataclass, field


@dataclass()
class Scoring():
    similarity_score: float
    is_similarity_match: bool = field(repr=False)
    scored_tag: Optional[str] = field(default=None, init=False)

    @property
    def is_match(self) -> bool:
        return self.is_similarity_match

    @property
    def score(self) -> float:
        return self.similarity_score


@dataclass()
class ScoringOptions:
    score_to_be_match: float = .5


class ScoringCalculator:

    def __init__(self, scoring_options: ScoringOptions = ScoringOptions()):
        self.scoring_options = scoring_options

    def __call__(self, embedding1: numpy.ndarray, embedding2: numpy.ndarray) -> Scoring:
        similarity_score = similarity_metric(embedding1, embedding2)
        return Scoring(similarity_score, similarity_score >= self.scoring_options.score_to_be_match)

    def describe(self) -> Dict[str, Any]:
        return {
            "scoring_options": self.scoring_options,
            "scoring": [
                "similarity_metric(embedding1, embedding2)"
            ]
        }
