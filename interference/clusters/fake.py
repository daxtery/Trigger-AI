from interference.clusters.processor import Processor
from typing import Any, Dict, List, Set

import numpy

class Fake(Processor):

    def __init__(self) -> None:
        self.tags: Set[str] = set()

    def update(self, tag: str, instance: numpy.ndarray) -> None:
        return

    def remove(self, tag: str) -> None:
        self.tags.remove(tag)

    def get_cluster_by_tag(self, tag: str) -> int:
        return 1

    def get_tags_in_cluster(self, cluster_id: int) -> List[str]:
        if cluster_id == 1:
            return list(self.tags)

        return []

    def get_cluster_ids(self) -> List[int]:
        return [1]

    def process(self, tag: str, instance: numpy.ndarray) -> None:
        self.tags.add(tag)

    def describe(self) -> Dict[str, Any]:
        """
        This describes this clustering algorithm's parameters
        """

        return {
            "name": "Fake",
            "parameters": {}
        }

    def safe_file_name(self) -> str:
        return f"Fake"

    def predict(self, instance: Any) -> int:
        return 1