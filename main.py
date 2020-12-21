from trigger.clusters.gturbo import GTurbo
from trigger.operations import AddInfo, CalculateMatchesInfo, Operation, OperationType
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

t = TestRunner(
    GTurbo,
    param_grid,
    [
        Operation(OperationType.ADD, AddInfo(tag="1", value=numpy.array([1, 1]))),
        Operation(OperationType.ADD, AddInfo(tag="2", value=numpy.array([1, 2]))),
        Operation(OperationType.ADD, AddInfo(tag="3", value=numpy.array([1, 3]))),
        Operation(OperationType.ADD, AddInfo(tag="4", value=numpy.array([1, 4]))),
        Operation(OperationType.CALCULATE_MATCHES, CalculateMatchesInfo(value=numpy.array([2, 2]), fetch_matched_value=True)),
    ]
)

t.run_tests()