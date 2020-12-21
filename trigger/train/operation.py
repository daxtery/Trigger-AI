from typing import Any, Generic, List, Optional, TypeVar

from dataclasses import dataclass
from enum import Enum, unique


@unique
class OperationType(Enum):
    ADD = 0
    UPDATE = 1
    REMOVE = 2
    CALCULATE_MATCHES = 3
    EVALUATE_CLUSTERS = 4
    EVALUATE_MATCHES = 5
    EVALUATE_CLUSTERS_AND_MATCHES = 6


T = TypeVar('T')


@dataclass()
class AddInfo(Generic[T]):
    tag: str
    value: T
    transformer_key: Optional[str] = None


@dataclass()
class UpdateInfo(AddInfo):
    pass


@dataclass()
class RemoveInfo():
    tag: str


@dataclass()
class CalculateMatchesInfo(Generic[T]):
    value: T
    transformer_key: Optional[str] = None


@dataclass()
class EvaluateClustersInfo():
    pass


@dataclass()
class EvaluateMatchesInfo():
    values: List[CalculateMatchesInfo]


@dataclass()
class EvaluateClustersAndMatchesInfo(EvaluateClustersInfo, EvaluateMatchesInfo):
    pass


OT = TypeVar('OT',
             AddInfo,
             RemoveInfo,
             CalculateMatchesInfo,
             EvaluateClustersInfo,
             EvaluateMatchesInfo,
             EvaluateClustersAndMatchesInfo
             )


@dataclass()
class Operation(Generic[OT]):
    type: OperationType
    info: OT
