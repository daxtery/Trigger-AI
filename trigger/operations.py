from typing import Generic, List, Optional, TypeVar

from dataclasses import dataclass
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
    EVALUATE_CLUSTERS_AND_MATCHES = 7


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
    fetch_instance: bool = False


@dataclass()
class EvaluateClustersAndMatchesInfo(EvaluateClustersInfo, EvaluateMatchesInfo):
    pass


OT = TypeVar('OT',
             AddInfo,
             RemoveInfo,
             UpdateInfo,
             CalculateScoringInfo,
             CalculateMatchesInfo,
             EvaluateClustersInfo,
             EvaluateMatchesInfo,
             EvaluateClustersAndMatchesInfo
             )


@dataclass()
class Operation(Generic[OT]):
    type: OperationType
    info: OT
