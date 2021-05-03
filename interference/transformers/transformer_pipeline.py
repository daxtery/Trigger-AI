from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy

T = TypeVar('T')


@dataclass
class Instance(Generic[T]):
    value: T
    embedding: numpy.ndarray = field(repr=False)


class TransformerPipeline(Generic[T]):

    @abstractmethod
    def calculate_embedding(self, value: T) -> numpy.ndarray: ...

    def transform(self, value: T) -> Instance[T]:
        embedding = self.calculate_embedding(value)
        return Instance(value, embedding)


class NumpyToInstancePipeline(TransformerPipeline[numpy.ndarray]):

    def calculate_embedding(self, value: numpy.ndarray):
        assert isinstance(value, numpy.ndarray)
        return value


class IdentityPipeline(TransformerPipeline[T]):

    def calculate_embedding(self, value: Instance[T]) -> numpy.ndarray:
        assert hasattr(value, 'embedding')
        return value.embedding

    def transform(self, value: Instance[T]) -> Instance[T]:
        embedding = self.calculate_embedding(value)
        assert hasattr(value, 'value')
        return Instance(value.value, embedding)
