import numpy

from interference.evaluation.match import eval_matches
from interference.evaluation.cluster import eval_cluster
from interference.scoring import ScoringCalculator, Scoring
from interference.operations import AddInfo, CalculateMatchesInfo, CalculateScoringInfo, EvaluateClustersInfo, EvaluateMatchesInfo, Operation, OperationType, RemoveInfo, UpdateInfo
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

    def on_operation_add(self, operation: Operation[AddInfo]):
        add_info: AddInfo = operation.info
        instance = self.try_create_instance_from_value(add_info.transformer_key, add_info.value)

        if instance is None:
            return None

        return self.add(add_info.tag, instance)

    def on_operation_update(self, operation: Operation[UpdateInfo]):
        update_info: UpdateInfo = operation.info

        instance = self.try_create_instance_from_value(update_info.transformer_key, update_info.value)

        if instance is None:
            return None

        return self.update(update_info.tag, instance)

    def on_operation_remove(self, operation: Operation[RemoveInfo]):
        remove_info: RemoveInfo = operation.info
        return self.remove(remove_info.tag)
    
    def on_operation_calculate_scores(self, operation: Operation[CalculateScoringInfo]):
        calculate_scoring_info: CalculateScoringInfo = operation.info

        instance = self.try_create_instance_from_value(calculate_scoring_info.transformer_key, calculate_scoring_info.value)

        if instance is None:
            return None

        return self.get_scorings_for(instance)
    
    def on_operation_calculate_matches(self, operation: Operation[CalculateMatchesInfo]):
        calculate_matches_info: CalculateMatchesInfo = operation.info

        instance = self.try_create_instance_from_value(calculate_matches_info.transformer_key, calculate_matches_info.value)

        if instance is None:
            return None

        return self.get_matches_for(instance)

    def _calculate_operation_matches_inner(self, values: Sequence[CalculateMatchesInfo]):

        all_instances: List[Instance] = []
        all_scorings: List[Sequence[Scoring]] = []

        for value_to_match in values:
            instance = self.try_create_instance_from_value(value_to_match.transformer_key, value_to_match.value)
            
            if instance is None:
                continue
        
            all_instances.append(instance)

            scorings = self.get_scorings_for(instance)

            all_scorings.append(scorings)
        
        return all_instances, all_scorings
    
    def _evaluate_matches_inner(self, values: Sequence[CalculateMatchesInfo]):

        instances, scorings = self._calculate_operation_matches_inner(values)

        return eval_matches(instances, scorings)
    
    def on_operation_evaluate_matches(self, operation: Operation[EvaluateMatchesInfo]):
        evaluate_matches_info: EvaluateMatchesInfo = operation.info

        evaluation = self._evaluate_matches_inner(evaluate_matches_info.values)

        if not evaluate_matches_info.fetch_instance:
            del evaluation["by_instance"]

        return evaluation
    
    def on_operation_evaluate_clusters(self, operation: Operation[EvaluateClustersInfo]):
        return eval_cluster(self)

    def on_operation(self, operation: Operation):
        
        if operation.type == OperationType.ADD: 
            return self.on_operation_add(operation)
        elif operation.type == OperationType.REMOVE:
            return self.on_operation_remove(operation)
        elif operation.type == OperationType.UPDATE:
            return self.on_operation_update(operation)
        elif operation.type == OperationType.CALCULATE_SCORES:
            return self.on_operation_calculate_scores(operation)
        elif operation.type == OperationType.CALCULATE_MATCHES:
            return self.on_operation_calculate_matches(operation)
        elif operation.type == OperationType.EVALUATE_CLUSTERS:
            return self.on_operation_evaluate_clusters(operation)
        elif operation.type == OperationType.EVALUATE_MATCHES:
            return self.on_operation_evaluate_matches(operation)
        else:
            pass


    def describe(self):
        return {
            "transformers": { key: transformer.__class__.__name__  for key, transformer in self.transformers.items() },
            "scoring_calculator": self.scoring_calculator.describe()
        }