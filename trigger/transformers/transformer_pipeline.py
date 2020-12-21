from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy

T = TypeVar('T')

@dataclass
class Instance(Generic[T]):
    value: T
    embedding: numpy.ndarray

class TransformerPipeline(ABC, Generic[T]):

    @abstractmethod
    def transform(self, value: T) -> Instance[T]:
        pass