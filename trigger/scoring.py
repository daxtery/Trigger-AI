from trigger.transformers.transformer_pipeline import Instance
from trigger.metrics.match import similarity_metric
from typing import Any, Dict, Generic, TypeVar

from dataclasses import dataclass, field

T = TypeVar('T')


@dataclass()
class Scoring(Generic[T]):
    scored_tag: str
    similarity_score: float

    is_similarity_match: bool = field(repr=False)

    @property
    def is_match(self) -> bool:
        return self.is_similarity_match


@dataclass()
class ScoringOptions:
    score_to_be_match: float = .5


class ScoringCalculator:

    def __init__(self, scoring_options: ScoringOptions = ScoringOptions()):
        self.scoring_options = scoring_options

    def __call__(self, instance1: Instance, tag: str, instance2: Instance) -> Scoring:
        similarity_score = similarity_metric(instance1.embedding, instance2.embedding)
        return Scoring(tag, similarity_score, similarity_score >= self.scoring_options.score_to_be_match)

    def describe(self) -> Dict[str, Any]:
        return {
            "scoring_options": self.scoring_options,
            "scoring": [
                "similarity_metric(instance1.embedding, instance2.embedding)"
            ]
        }
