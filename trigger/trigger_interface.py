from trigger.evaluation.match import eval_matches
from trigger.evaluation.cluster import eval_cluster
from trigger.scoring import ScoringCalculator, Scoring
from trigger.operations import AddInfo, CalculateMatchesInfo, CalculateScoringInfo, EvaluateClustersAndMatchesInfo, EvaluateClustersInfo, EvaluateMatchesInfo, Operation, OperationType, RemoveInfo, UpdateInfo
from trigger.transformers.transformer_pipeline import Instance, TransformerPipeline
from trigger.clusters.processor import Processor

from typing import Any, Dict, List, Optional, Tuple

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('trigger_interface')
logger.setLevel(logging.INFO)

class TriggerInterface:
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

    def find_transformer_for_key(self, key: str) -> Optional[TransformerPipeline]:
        transformer = self.transformers.get(key, None)

        if transformer is None:
            logger.warning(f"No transformer with key='{key}'")
        
        return transformer

    def create_instance_or_none(self, transformer_key: str, value: Any) -> Optional[Instance]:
        transformer = self.find_transformer_for_key(transformer_key)

        if transformer is None:
            # TODO: LOG THIS
            return None

        return transformer.transform(value)

    def add(self, tag: str, transformer_key: str, value: Any) -> bool:
        instance = self.create_instance_or_none(transformer_key, value)
        
        if instance is None:
            return False

        self.add_instance(tag, instance)
        return True


    def add_instance(self, tag: str, instance: Instance) -> None:
        self.processor.process(tag, instance.embedding)
        self.instances_map[tag] = instance

    def update(self, tag: str, transformer_key: str, value: Any) -> bool:
        instance = self.create_instance_or_none(transformer_key, value)

        if instance is None:
            return False
        
        return self.update_instance(tag, instance)

    def update_instance(self, tag: str, instance: Instance) -> bool:
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

    def get_scorings_for(self, transformer_key: str, value: Any) -> Optional[List[Scoring]]:
        instance = self.create_instance_or_none(transformer_key, value)

        if instance is None:
            return None

        return self.get_scorings_for_instance(instance)

    def get_scorings_for_instance(self, instance: Instance) -> List[Scoring]:
        
        if len(self.instances_map) == 0:
            return []

        would_be_cluster_id = self.processor.predict(instance.embedding)

        tags = self.processor.get_tags_in_cluster(would_be_cluster_id)

        temp = [
            self.calculate_scoring_between_instance_and_tag_or_none(instance, tag)
            for tag in tags
        ]

        return list(filter(None, temp))

    
    def get_matches_for(self, transformer_key: str, value: Any) -> Optional[List[Scoring]]:
        instance = self.create_instance_or_none(transformer_key, value)

        if instance is None:
            return None

        return self.get_matches_for_instance(instance)
    
    def get_matches_for_instance(self, instance: Instance) -> List[Scoring]:

        scorings = self.get_scorings_for_instance(instance)

        return [
            scoring
            for scoring in scorings
            if scoring.is_match
        ]

    def calculate_scoring_between_value_and_tag(self, transformer_key: str, value: Any, tag: str) -> Optional[Scoring]:
        instance = self.create_instance_or_none(transformer_key, value)

        if instance is None:
            return None

        return self.calculate_scoring_between_instance_and_tag_or_none(instance, tag)

    def calculate_scoring_between_instance_and_tag_or_none(self, instance: Instance, tag: str) -> Optional[Scoring]:
        instances = self.get_instances_by_tag([tag])

        if len(instances) == 0:
            # TODO: LOG THIS
            return None

        instance2 = instances[0]

        return self.calculate_scoring_between_instances(instance, tag, instance2)

    def calculate_scoring_between_instances(self, instance1: Instance, tag2: str, instance2: Instance) -> Scoring:
        return self.scoring_calculator(instance1, tag2, instance2)
    
    
    def get_instances_by_tag(self, tags: List[str]) -> List[Instance]:
        temp = [
            self.instances_map.get(tag, None)
            for tag in tags
        ]

        return list(filter(None, temp))

    def on_operation_add(self, operation: Operation):
        add_info: AddInfo = operation.info
        return self.add(add_info.tag, add_info.transformer_key, add_info.value)

    def on_operation_update(self, operation: Operation):
        remove_info: RemoveInfo = operation.info
        return self.remove(remove_info.tag)

    def on_operation_remove(self, operation: Operation):
        update_info: UpdateInfo = operation.info
        return self.update(update_info.tag, update_info.transformer_key, update_info.value)
    
    def on_operation_calculate_scores(self, operation: Operation):
        calculate_scoring_info: CalculateScoringInfo = operation.info
        return self.get_scorings_for(
            calculate_scoring_info.transformer_key,
            calculate_scoring_info.value,
        )
    
    def on_operation_calculate_matches(self, operation: Operation):
        calculate_matches_info: CalculateMatchesInfo = operation.info
        return self.get_matches_for(calculate_matches_info.transformer_key, calculate_matches_info.value)

    def _calculate_operation_matches_inner(self, values: List[CalculateMatchesInfo]) -> \
            Tuple[List[Instance], List[List[Scoring]]]:

        all_instances: List[Instance] = []
        all_scorings: List[List[Scoring]] = []

        for value_to_match in values:
            instance = self.create_instance_or_none(value_to_match.transformer_key, value_to_match.value)
            if instance is None:
                continue
        
            all_instances.append(instance)

            scorings = self.get_scorings_for_instance(instance)

            all_scorings.append(scorings)
        
        return all_instances, all_scorings
    
    def on_operation_evaluate_matches(self, operation: Operation):
        evaluate_matches_info: EvaluateMatchesInfo = operation.info

        instances, scorings = self._calculate_operation_matches_inner(evaluate_matches_info.values)

        evaluation = eval_matches(instances, scorings)

        if not evaluate_matches_info.fetch_instance:
            del evaluation["by_instance"]

        return evaluation

    def on_operation_evaluate_clusters_and_matches(self, operation: Operation):
        evaluate_clusters_and_matches_info: EvaluateClustersAndMatchesInfo = operation.info
        clusters_evaluation = eval_cluster(self)

        instances, scorings = self._calculate_operation_matches_inner(evaluate_clusters_and_matches_info.values)

        matches_evaluation = eval_matches(instances, scorings)

        if not evaluate_clusters_and_matches_info.fetch_instance:
            del matches_evaluation["by_value"]

        results = clusters_evaluation
        results['matches_results'] = matches_evaluation

        return results
    
    def on_operation_evaluate_clusters(self, operation: Operation):
        evaluate_clusters_info: EvaluateClustersInfo = operation.info
        return eval_cluster(self)

    def on_operation(self, operation: Operation):
        
        if operation.type == OperationType.ADD: 
            return self.on_operation_add(operation)
        if operation.type == OperationType.REMOVE:
            return self.on_operation_remove(operation)
        if operation.type == OperationType.UPDATE:
            return self.on_operation_update(operation)
        if operation.type == OperationType.CALCULATE_SCORES:
            return self.on_operation_calculate_scores(operation)
        if operation.type == OperationType.CALCULATE_MATCHES:
            return self.on_operation_calculate_matches(operation)
        if operation.type == OperationType.EVALUATE_CLUSTERS:
            return self.on_operation_evaluate_clusters(operation)
        if operation.type == OperationType.EVALUATE_MATCHES:
            return self.on_operation_evaluate_matches(operation)
        if operation.type == OperationType.EVALUATE_CLUSTERS_AND_MATCHES:
            return self.on_operation_evaluate_clusters_and_matches(operation)


    def describe(self) -> Dict[str, Any]:
        return {
            "transformers": { key: transformer.__class__.__name__  for key, transformer in self.transformers.items() },
            "scoring_calculator": self.scoring_calculator.describe()
        }