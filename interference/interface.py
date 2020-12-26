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
U = TypeVar('U')


class Interface:
    def __init__(
        self,
        processor: Processor,
        transformers: Dict[str, TransformerPipeline],
        scoring_calculator: ScoringCalculator = ScoringCalculator()
    ) -> None:
        self.processor: Processor = processor
        self.transformers: Dict[str, TransformerPipeline] = transformers
        self.scoring_calculator = scoring_calculator
        self.instances_map: Dict[str, Instance] = {}

    def try_get_transformer_for_key(self, key: str) -> Optional[TransformerPipeline]:
        return self.transformers.get(key, None)

    def try_create_instance_from_value(self, key: str, value: T) -> Optional[Instance[T]]:
        transformer = self.try_get_transformer_for_key(key)

        if transformer is None:
            return None

        transformer = cast(TransformerPipeline[T], transformer)

        return transformer.transform(value)

    def add(self, tag: str, instance: Instance[T]) -> None:
        self.processor.process(tag, instance.embedding)
        self.instances_map[tag] = instance

    def update(self, tag: str, instance: Instance[T]) -> bool:
        if not tag in self.instances_map:
            return False
        
        self.processor.update(tag, instance.embedding)
        self.instances_map[tag] = instance

        return True

    def remove(self, tag: str) -> bool:
        if not tag in self.instances_map:
            return False

        self.processor.remove(tag)
        del self.instances_map[tag]
        return True

    def get_scorings_for(self, instance: Instance[T]) -> Sequence[Scoring]:
        
        if len(self.instances_map) == 0:
            return []

        would_be_cluster_id = self.processor.predict(instance.embedding)

        tags = self.processor.get_tags_in_cluster(would_be_cluster_id)
        instances = [ self.instances_map[tag] for tag in tags ]

        scorings: List[Scoring] = []

        for tag2, instance2 in zip(tags, instances):
            scoring = self.calculate_scoring_between_instances(instance, instance2)
            scoring.scored_tag = tag2
            scorings.append(scoring)

        return scorings

    
    def get_matches_for(self, instance: Instance[T]) -> Sequence[Scoring]:

        scorings = self.get_scorings_for(instance)

        return [
            scoring
            for scoring in scorings
            if scoring.is_match
        ]

    def calculate_scoring_between_instances(self, instance1: Instance[T], instance2: Instance[U]) -> Scoring:
        return self.scoring_calculator(instance1, instance2)
    
    
    def get_instances_by_tag(self, tags: Sequence[str]) -> Sequence[Instance]:
        temp = [
            self.instances_map.get(tag, None)
            for tag in tags
        ]

        return list(filter(None, temp))

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

    def _calculate_operation_matches_inner(self, values: Sequence[CalculateMatchesInfo]) -> \
            Tuple[Sequence[Instance], Sequence[Sequence[Scoring]]]:

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
        evaluate_clusters_info: EvaluateClustersInfo = operation.info
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


    def describe(self) -> Dict[str, Any]:
        return {
            "transformers": { key: transformer.__class__.__name__  for key, transformer in self.transformers.items() },
            "scoring_calculator": self.scoring_calculator.describe()
        }