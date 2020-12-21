from abc import ABC, abstractmethod
from typing import Any, List, Optional, Dict

import numpy


class Processor(ABC):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def process(self, tag: str, instance: numpy.ndarray) -> None:
        pass

    @abstractmethod
    def update(self, tag: str, instance: numpy.ndarray) -> None:
        pass

    @abstractmethod
    def remove(self, tag: str) -> None:
        pass

    @abstractmethod
    def get_cluster_by_tag(self, tag: str) -> Optional[int]:
        pass

    @abstractmethod
    def get_tags_in_cluster(self, cluster_id: int) -> List[str]:
        pass

    @abstractmethod
    def get_cluster_ids(self) -> List[int]:
        pass

    @abstractmethod
    def predict(self, instance: numpy.ndarray) -> int:
        pass

    @abstractmethod
    def describe(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def safe_file_name(self) -> str:
        pass