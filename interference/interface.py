import numpy


from interference.scoring import ScoringCalculator, Scoring
from interference.transformers.transformer_pipeline import Instance, TransformerPipeline
from interference.clusters.processor import Processor

from typing import Any, Dict, List, Optional, Tuple, TypeVar, cast, Sequence

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('interface')
logger.setLevel(logging.INFO)

T = TypeVar('T')


class Interface:
    def __init__(
        self,
        processor: Processor,
        transformers: Dict[str, TransformerPipeline],
        scoring_calculator: ScoringCalculator
    ) -> None:
        self.processor = processor
        self.transformers = transformers
        self.scoring_calculator = scoring_calculator
        self.embeddings_map: Dict[str, numpy.ndarray] = {}

    def try_get_transformer_for_key(self, key: str):
        return self.transformers.get(key, None)

    def try_create_instance_from_value(self, key: str, value: T):
        transformer = self.try_get_transformer_for_key(key)

        if transformer is None:
            return None

        transformer = cast(TransformerPipeline[T], transformer)

        return transformer.transform(value)

    def add(self, tag: str, instance: Instance):
        self.processor.process(tag, instance.embedding)
        self.embeddings_map[tag] = instance.embedding

    def update(self, tag: str, instance: Instance):
        if not tag in self.embeddings_map:
            return False
        
        self.processor.update(tag, instance.embedding)
        self.embeddings_map[tag] = instance.embedding

        return True

    def remove(self, tag: str):
        if not tag in self.embeddings_map:
            return False

        self.processor.remove(tag)
        del self.embeddings_map[tag]
        return True

    def get_scorings_for(self, instance: Instance):
        
        if len(self.embeddings_map) == 0:
            return []

        would_be_cluster_id = self.processor.predict(instance.embedding)

        tags = self.processor.get_tags_in_cluster(would_be_cluster_id)
        embeddings = [ self.embeddings_map[tag] for tag in tags ]

        scorings: List[Scoring] = []

        for tag2, embedding2 in zip(tags, embeddings):
            scoring = self.calculate_scoring_between_embeddings(instance.embedding, embedding2)
            scoring.scored_tag = tag2
            scorings.append(scoring)

        return scorings

    
    def get_matches_for(self, instance: Instance):
        scorings = self.get_scorings_for(instance)

        return [
            scoring
            for scoring in scorings
            if scoring.is_match
        ]


    def calculate_scoring_between_instances(self, instance1: Instance, instance2: Instance):
        return self.calculate_scoring_between_embeddings(instance1.embedding, instance2.embedding)


    def calculate_scoring_between_embeddings(self, embedding1: numpy.ndarray, embedding2: numpy.ndarray):
        return self.scoring_calculator(embedding1, embedding2)


    def get_embeddings_by_tag(self, tags: Sequence[str]):
        return [
            self.embeddings_map[tag]
            for tag in tags
            if tag in self.embeddings_map
        ]

    def describe(self):
        return {
            "transformers": { key: transformer.__class__.__name__  for key, transformer in self.transformers.items() },
            "scoring_calculator": self.scoring_calculator.describe()
        }