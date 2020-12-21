from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional, Dict

import numpy


class Processor(ABC):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def process(self, tag: str, instance: numpy.ndarray, custom_data: Any = None) -> None:
        pass

    @abstractmethod
    def update(self, tag: str, instance: numpy.ndarray, custom_data: Any = None) -> None:
        pass

    @abstractmethod
    def remove(self, tag: str) -> None:
        pass

    @abstractmethod
    def get_cluster_by_tag(self, tag: str) -> Optional[int]:
        pass

    @abstractmethod
    def get_custom_data_by_tag(self, tag: str) -> Optional[Any]:
        pass

    @abstractmethod
    def get_instance_by_tag(self, tag: str) -> Optional[numpy.ndarray]:
        pass

    @abstractmethod
    def get_instances_and_tags_in_cluster(self, cluster_id: int) -> Tuple[List[numpy.ndarray], List[str]]:
        pass

    # TODO: It's debatable if this is needed.
    # Mainly for Debugging?
    @abstractmethod
    def get_all_instances_with_tags(self) -> Tuple[List[numpy.ndarray], List[str]]:
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

    @abstractmethod
    def compute_cluster_score(self) -> float:
        pass
