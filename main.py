import numpy
from trigger.train.operation import AddInfo, Operation, OperationType
from trigger.test.test_operations_runner import TestRunner
from trigger.train.cluster.ecm.ecm import ECM
from trigger.train.trigger_interface import TriggerInterface

ecm = ECM(distance_threshold=0.5)

interface = TriggerInterface(ecm, {})

param_grid = {
    "distance_threshold": [0.4, 0.5, 0.6, 0.7, 0.8]
}

t = TestRunner(
    ECM,
    param_grid,
    [
        Operation(OperationType.ADD, AddInfo(tag="1", value=numpy.array([1, 1]))),
        Operation(OperationType.ADD, AddInfo(tag="2", value=numpy.array([1, 2]))),
        Operation(OperationType.ADD, AddInfo(tag="3", value=numpy.array([1, 3]))),
        Operation(OperationType.ADD, AddInfo(tag="4", value=numpy.array([1, 4]))),
    ]
)

t.run_tests()