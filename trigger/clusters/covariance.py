from trigger.clusters.processor import Processor
import numpy as np
from typing import Any, Dict, List, Tuple
from scipy.spatial.distance import mahalanobis

class ClusterNode:

    def __init__(self, id, instance, initial_std) -> None:

        self.id = id
        self.cov_matrix = np.eye(1024)
        self.std = initial_std
        self.mean = instance
        self.instances = [instance]

        i_observation = instance.reshape((1024, 1))

        self.observations = i_observation

    def add_instance(self, instance) -> None:

        self.instances.append(instance)

        self.observations = np.hstack([self.observations, instance.reshape((1024, 1))])

        self.cov_matrix = np.cov(self.observations)

        self.mean = np.mean(self.instances, axis=0)

        std_vector = np.std(self.instances, axis=0)

        self.std = np.linalg.norm(std_vector)


class CovarianceCluster(Processor):

    def __init__(self, initial_std=0.01) -> None:

        self.initial_std = initial_std
        self.tag_to_cluster: Dict[str, int] = {}
        self.id = 0
        self.clusters: Dict[int, ClusterNode] = {}

    def add_to_cluster(self, tag, instance) -> None:

        id = -1

        if len(self.clusters) == 0:

            id = self._create_node(instance)

        else:

            distance, node = self.brute_search(instance)

            if distance < node.std:

                node.add_instance(instance)
                id = node.id

            else:

                id = self._create_node(instance)

        self.tag_to_cluster[tag] = id

    def remove_from_cluster(self, tag) -> None:
        
        pass

    def stat_distance(self, instance, node: ClusterNode) -> float:

        return mahalanobis(instance, node.mean, node.cov_matrix)

    def brute_search(self, instance) -> Tuple[float, ClusterNode]:

        nodes = list(self.clusters.values())

        curr_node = nodes[0]
        distance = self.stat_distance(instance, nodes[0])

        for node in nodes[1:]:

            c_distance = self.stat_distance(instance, node)

            if c_distance < distance:

                distance = c_distance
                curr_node = node

        return (distance, curr_node)

    def _create_node(self, instance) -> int:

        id = self.id
        self.id += 1

        new_node = ClusterNode(id, instance, self.initial_std)

        self.clusters[id] = new_node

        return id

    def process(self, tag: str, instance: np.ndarray) -> None:

        self.add_to_cluster(tag, instance)

    def update(self, tag: str, instance: np.ndarray) -> None:

        self.remove(tag)

        self.process(tag, instance)

    def remove(self, tag: str) -> None:

        self.remove_from_cluster(tag)

    def get_cluster_by_tag(self, tag: str) -> int:

        return self.tag_to_cluster[tag]

    def get_tags_in_cluster(self, cluster_id: int) -> List[str]:

        return [tag for tag, id in self.tag_to_cluster.items() if id ==
                cluster_id]

    def get_cluster_ids(self) -> List[int]:
        
        return [
            id for id
            in self.clusters.keys()
        ]

    def predict(self, instance: np.ndarray) -> int:

        return self.brute_search(instance)[1].id

    def describe(self) -> Dict[str, Any]:

        return {
            "name": "Covariance Cluster",
            "parameters": {
                "initial_std": self.initial_std
            }
        }

    def safe_file_name(self) -> str:

        return f"CovCluster = initial_std={self.initial_std}"