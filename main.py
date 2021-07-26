from interference.clusters.gturbo import GTurbo
from interference.test.operations import AddInfo, CalculateMatchesInfo, CalculateScoringInfo, EvaluateClustersInfo, EvaluateMatchesInfo, Operation, OperationType, RemoveInfo, UpdateInfo
from interference.test.test_operations_runner import TestRunner

import numpy

param_grid = {
    "epsilon_b": [0.01],
    "epsilon_n": [0],
    "lam": [500],
    "beta": [0.9995],
    "alpha": [0.95],
    "max_age": [500],
    "r0": [0.5, 1, 2.5]
}

cm = CalculateMatchesInfo(value=numpy.array([2, 2]))

ev = EvaluateMatchesInfo(values=[
    cm
], fetch_instance=True)

t = TestRunner(
    GTurbo,
    param_grid,
    [
        Operation(OperationType.ADD, AddInfo(tag="1", value=numpy.array([1, 1]))),
        Operation(OperationType.ADD, AddInfo(tag="2", value=numpy.array([1, 2]))),
        Operation(OperationType.ADD, AddInfo(tag="3", value=numpy.array([1, 3]))),
        Operation(OperationType.ADD, AddInfo(tag="4", value=numpy.array([1, 4]))),
        Operation(OperationType.UPDATE, UpdateInfo(tag="4", value=numpy.array([18, 16]))),
        Operation(OperationType.REMOVE, RemoveInfo(tag="4")),
        Operation(OperationType.ADD, AddInfo(tag="4", value=numpy.array([18, 16]))),
        Operation(OperationType.CALCULATE_SCORES, info = CalculateScoringInfo(value=numpy.array([2, 2]))),
        Operation(OperationType.CALCULATE_MATCHES, info = cm),
        Operation(OperationType.EVALUATE_CLUSTERS, EvaluateClustersInfo()),
        Operation(OperationType.EVALUATE_MATCHES, ev),
    ],
)

t.run_tests()