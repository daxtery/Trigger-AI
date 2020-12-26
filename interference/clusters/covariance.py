from interference.clusters.processor import Processor
import numpy as np
from typing import Any, Dict, Sequence, Tuple
from scipy.spatial.distance import mahalanobis

class ClusterNode:

    def __init__(self, id, embedding: np.ndarray, initial_std: float, dimensions: int) -> None:

        self.id = id
        self.dimensions = dimensions

        self.cov_matrix = np.eye(dimensions)
        self.std = initial_std
        self.mean = embedding
        self.instances = [embedding]

        i_observation = embedding.reshape((dimensions, 1))

        self.observations = i_observation

    def add_embedding(self, embedding) -> None:

        self.instances.append(embedding)

        self.observations = np.hstack([self.observations, embedding.reshape((self.dimensions, 1))])

        self.cov_matrix = np.cov(self.observations)

        self.mean = np.mean(self.instances, axis=0)

        std_vector = np.std(self.instances, axis=0)

        self.std = np.linalg.norm(std_vector)


class CovarianceCluster(Processor):

    def __init__(self, dimensions: int, initial_std: float = 0.01) -> None:

        self.initial_std = initial_std
        self.tag_to_cluster: Dict[str, int] = {}
        self.id = 0
        self.clusters: Dict[int, ClusterNode] = {}
        self.dimensions = dimensions

    def add_to_cluster(self, tag: str, embedding: np.ndarray) -> None:

        id = -1

        if len(self.clusters) == 0:

            id = self._create_node(embedding)

        else:

            distance, node = self.brute_search(embedding)

            if distance < node.std:

                node.add_embedding(embedding)
                id = node.id

            else:

                id = self._create_node(embedding)

        self.tag_to_cluster[tag] = id

    def remove_from_cluster(self, tag: str) -> None:
        
        pass

    def stat_distance(self, embedding: np.ndarray, node: ClusterNode) -> float:

        return mahalanobis(embedding, node.mean, node.cov_matrix)

    def brute_search(self, embedding: np.ndarray) -> Tuple[float, ClusterNode]:

        nodes = list(self.clusters.values())

        curr_node = nodes[0]
        distance = self.stat_distance(embedding, nodes[0])

        for node in nodes[1:]:

            c_distance = self.stat_distance(embedding, node)

            if c_distance < distance:

                distance = c_distance
                curr_node = node

        return (distance, curr_node)

    def _create_node(self, embedding: np.ndarray) -> int:

        id = self.id
        self.id += 1

        new_node = ClusterNode(id, embedding, self.initial_std, self.dimensions)

        self.clusters[id] = new_node

        return id

    def process(self, tag: str, embedding: np.ndarray) -> None:

        self.add_to_cluster(tag, embedding)

    def update(self, tag: str, embedding: np.ndarray) -> None:

        self.remove(tag)

        self.process(tag, embedding)

    def remove(self, tag: str) -> None:

        self.remove_from_cluster(tag)

    def get_cluster_by_tag(self, tag: str) -> int:

        return self.tag_to_cluster[tag]

    def get_tags_in_cluster(self, cluster_id: int) -> Sequence[str]:

        return [tag for tag, id in self.tag_to_cluster.items() if id ==
                cluster_id]

    def get_cluster_ids(self) -> Sequence[int]:
        
        return [
            id for id
            in self.clusters.keys()
        ]

    def predict(self, embedding: np.ndarray) -> int:

        return self.brute_search(embedding)[1].id

    def describe(self) -> Dict[str, Any]:

        return {
            "name": "Covariance Cluster",
            "parameters": {
                "initial_std": self.initial_std
            }
        }

    def safe_file_name(self) -> str:

        return f"CovCluster = initial_std={self.initial_std}"