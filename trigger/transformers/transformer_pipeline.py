from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar
from typing_extensions import Protocol

import numpy

T = TypeVar('T')

@dataclass
class Instance(Generic[T]):
    value: T
    embedding: numpy.ndarray

class TransformerPipeline(Protocol, Generic[T]):

    @abstractmethod
    def transform(self, value: T) -> Instance[T]:...