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


class NumpyToInstancePipeline(TransformerPipeline[numpy.ndarray]):

    def transform(self, value: numpy.ndarray) -> Instance[numpy.ndarray]:
        assert isinstance(value, numpy.ndarray)
        return Instance(value, value)
    
class IdentityPipeline(TransformerPipeline[T]):

    def transform(self, value: Instance[T]) -> Instance[T]:
        assert value.embedding
        assert value.value
        return value