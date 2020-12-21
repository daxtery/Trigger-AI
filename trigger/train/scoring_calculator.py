import inspect
from inspect import signature
from trigger.train.scoring import Scoring
from trigger.metrics.match import similarity_metric
from typing import Any, Dict
from trigger.train.scoring_options import ScoringOptions
from trigger.train.transformers.transformer_pipeline import Instance


class ScoringCalculator:

    def __init__(self, scoring_options: ScoringOptions = ScoringOptions()):
        self.scoring_options = scoring_options

    def __call__(self, instance1: Instance, tag2: str, instance2: Instance) -> Scoring:
        similarity_score = similarity_metric(instance1.embedding, instance2.embedding)
        return Scoring(tag2, similarity_score, similarity_score >= self.scoring_options.score_to_be_match)

    def describe(self) -> Dict[str, Any]:
        return {
            "scoring_options": self.scoring_options,
            "scoring": [
                "similarity_metric(instance1.embedding, instance2.embedding)"
            ]
        }
