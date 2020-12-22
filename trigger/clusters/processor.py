from abc import ABC, abstractmethod
from typing import Any, List, Dict
from typing_extensions import Protocol

import numpy

class Processor(Protocol):

    @abstractmethod
    def process(self, tag: str, instance: numpy.ndarray) -> None:...

    @abstractmethod
    def update(self, tag: str, instance: numpy.ndarray) -> None:...

    @abstractmethod
    def remove(self, tag: str) -> None:...

    @abstractmethod
    def get_cluster_by_tag(self, tag: str) -> int:...

    @abstractmethod
    def get_tags_in_cluster(self, cluster_id: int) -> List[str]:...

    @abstractmethod
    def get_cluster_ids(self) -> List[int]:...

    @abstractmethod
    def predict(self, instance: numpy.ndarray) -> int:...

    @abstractmethod
    def describe(self) -> Dict[str, Any]:...

    @abstractmethod
    def safe_file_name(self) -> str:...