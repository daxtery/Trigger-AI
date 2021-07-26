from typing import List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from interference.scoring import Scoring
    from interference.transformers.transformer_pipeline import Instance
    from interference.interface import Interface

from interference.evaluation.match import eval_matches
from interference.evaluation.cluster import eval_cluster
from interference.test.operations import AddInfo, CalculateMatchesInfo, CalculateScoringInfo, EvaluateClustersInfo, EvaluateMatchesInfo, Operation, OperationType, RemoveInfo, UpdateInfo


def on_operation_add(interface: "Interface", operation: Operation[AddInfo]):
    add_info = operation.info
    instance = interface.try_create_instance_from_value(add_info.transformer_key, add_info.value)

    if instance is None:
        return None

    return interface.add(add_info.tag, instance)

def on_operation_update(interface: "Interface", operation: Operation[UpdateInfo]):
    update_info = operation.info

    instance = interface.try_create_instance_from_value(update_info.transformer_key, update_info.value)

    if instance is None:
        return None

    return interface.update(update_info.tag, instance)

def on_operation_remove(interface: "Interface", operation: Operation[RemoveInfo]):
    remove_info = operation.info
    return interface.remove(remove_info.tag)

def on_operation_calculate_scores(interface: "Interface", operation: Operation[CalculateScoringInfo]):
    calculate_scoring_info = operation.info
    instance = interface.try_create_instance_from_value(calculate_scoring_info.transformer_key, calculate_scoring_info.value)

    if instance is None:
        return None

    return interface.get_scorings_for(instance)

def on_operation_calculate_matches(interface: "Interface", operation: Operation[CalculateMatchesInfo]):
    calculate_matches_info = operation.info

    instance = interface.try_create_instance_from_value(calculate_matches_info.transformer_key, calculate_matches_info.value)

    if instance is None:
        return None

    return interface.get_matches_for(instance)

def _calculate_operation_matches_inner(interface: "Interface", values: Sequence[CalculateMatchesInfo]):

    all_instances: "List[Instance]" = []
    all_scorings: "List[Sequence[Scoring]]" = []

    for value_to_match in values:
        instance = interface.try_create_instance_from_value(value_to_match.transformer_key, value_to_match.value)
        
        if instance is None:
            continue
    
        all_instances.append(instance)

        scorings = interface.get_scorings_for(instance)

        all_scorings.append(scorings)
    
    return all_instances, all_scorings

def _evaluate_matches_inner(interface: "Interface", values: Sequence[CalculateMatchesInfo]):

    instances, scorings = _calculate_operation_matches_inner(interface, values)

    return eval_matches(instances, scorings)

def on_operation_evaluate_matches(interface: "Interface", operation: Operation[EvaluateMatchesInfo]):
    evaluate_matches_info = operation.info

    evaluation = _evaluate_matches_inner(interface, evaluate_matches_info.values)

    if not evaluate_matches_info.fetch_instance:
        del evaluation["by_instance"]

    return evaluation

def on_operation_evaluate_clusters(interface: "Interface", operation: Operation[EvaluateClustersInfo]):
    return eval_cluster(interface)


def on_operation(interface: "Interface", operation: Operation):
    
    if operation.type == OperationType.ADD: 
        return on_operation_add(interface, operation)
    elif operation.type == OperationType.REMOVE:
        return on_operation_remove(interface, operation)
    elif operation.type == OperationType.UPDATE:
        return on_operation_update(interface, operation)
    elif operation.type == OperationType.CALCULATE_SCORES:
        return on_operation_calculate_scores(interface, operation)
    elif operation.type == OperationType.CALCULATE_MATCHES:
        return on_operation_calculate_matches(interface, operation)
    elif operation.type == OperationType.EVALUATE_CLUSTERS:
        return on_operation_evaluate_clusters(interface, operation)
    elif operation.type == OperationType.EVALUATE_MATCHES:
        return on_operation_evaluate_matches(interface, operation)
    else:
        pass