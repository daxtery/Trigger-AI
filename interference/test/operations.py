from typing import Generic, List, TypeVar

from dataclasses import dataclass, field
from enum import Enum, unique


@unique
class OperationType(Enum):
    ADD = 0
    UPDATE = 1
    REMOVE = 2
    CALCULATE_MATCHES = 3
    CALCULATE_SCORES = 4
    EVALUATE_CLUSTERS = 5
    EVALUATE_MATCHES = 6


T = TypeVar('T')


@dataclass()
class AddInfo(Generic[T]):
    tag: str
    value: T
    transformer_key: str = "numpy"

UpdateInfo = AddInfo

@dataclass()
class RemoveInfo():
    tag: str


@dataclass()
class CalculateScoringInfo(Generic[T]):
    value: T
    transformer_key: str = "numpy"


CalculateMatchesInfo = CalculateScoringInfo

@dataclass()
class EvaluateClustersInfo():
    pass


@dataclass()
class EvaluateMatchesInfo():
    values: List[CalculateMatchesInfo]
    fetch_instance: bool = field(default=True, repr=False)



OT = TypeVar('OT',
             AddInfo,
             RemoveInfo,
             UpdateInfo,
             CalculateScoringInfo,
             CalculateMatchesInfo,
             EvaluateClustersInfo,
             EvaluateMatchesInfo,
             )


@dataclass()
class Operation(Generic[OT]):
    type: OperationType
    info: OT

