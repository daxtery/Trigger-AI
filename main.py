from trigger.clusters.covariance import CovarianceCluster
from trigger.clusters.ecm import ECM
from trigger.clusters.gturbo import GTurbo
from trigger.operations import AddInfo, CalculateMatchesInfo, CalculateScoringInfo, EvaluateClustersAndMatchesInfo, EvaluateClustersInfo, EvaluateMatchesInfo, Operation, OperationType
from trigger.test.test_operations_runner import TestRunner

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

fantastic = Operation(
    OperationType.EVALUATE_MATCHES,
    ev
)

t = TestRunner(
    GTurbo,
    param_grid,
    [
        Operation(OperationType.ADD, AddInfo(tag="1", value=numpy.array([1, 1]))),
        Operation(OperationType.ADD, AddInfo(tag="2", value=numpy.array([1, 2]))),
        Operation(OperationType.ADD, AddInfo(tag="3", value=numpy.array([1, 3]))),
        Operation(OperationType.ADD, AddInfo(tag="4", value=numpy.array([1, 4]))),
        Operation(OperationType.CALCULATE_SCORES, info = CalculateScoringInfo(value=numpy.array([2, 2]))),
        Operation(OperationType.CALCULATE_MATCHES, info = cm),
        Operation(OperationType.EVALUATE_CLUSTERS, EvaluateClustersInfo()),
        fantastic,
        Operation(OperationType.EVALUATE_CLUSTERS_AND_MATCHES, EvaluateClustersAndMatchesInfo(values=[cm], fetch_instance=True)),
    ]
)

t.run_tests()