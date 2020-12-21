import numpy as np

from trigger.train.cluster.Processor import Processor
from trigger.train.cluster.covariance.cluster_node import ClusterNode
from typing import Any, Dict, List, Optional, Tuple
from scipy.spatial.distance import mahalanobis

class CovarianceCluster(Processor):

    def __init__(self, initial_std=0.01) -> None:

        self.initial_std = initial_std
        self.instance_to_cluster: Dict[str, id] = {}
        self.id = 0
        self.clusters: Dict[int, ClusterNode] = {}
        self.tag_to_data = {}
        self.intances = {}

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

        self.instance_to_cluster[tag] = id
        self.intances[tag] = instance

    def remove_from_cluster(self, tag) -> None:

        pass


    def stat_distance(self, instance, node: ClusterNode):

        return mahalanobis(instance, node.mean, node.cov_matrix)

    def brute_search(self, instance) -> ClusterNode:

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

    def process(self, tag: str, instance: np.ndarray, custom_data: Any = None) -> None:

        self.add_to_cluster(tag, instance)
        self.tag_to_data[tag] = custom_data

    def update(self, tag: str, instance: np.ndarray, custom_data: Any = None) -> None:

        self.remove(tag)

        self.process(tag, instance, custom_data)

    def remove(self, tag: str) -> None:

        self.remove_from_cluster(tag)

    def get_cluster_by_tag(self, tag: str) -> Optional[int]:

        return self.instance_to_cluster[tag]

    def get_custom_data_by_tag(self, tag: str) -> Optional[Any]:

        return self.tag_to_data[tag]

    def get_instance_by_tag(self, tag: str) -> Optional[np.ndarray]:

        return self.instances.get(tag, None)

    def get_instances_and_tags_in_cluster(self, cluster_id: int) -> Tuple[List[np.ndarray], List[str]]:

        tags = [tag for tag, id in self.instance_to_cluster.items() if id ==
                cluster_id]
        instances = [self.instances[tag] for tag in tags]

        return (instances, tags)

    def get_all_instances_with_tags(self) -> Tuple[List[np.ndarray], List[str]]:

        instances = [instance for instance in self.instances.values()]
        tags = [tag for tag in self.instances.keys()]

        return (instances, tags)

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

    def compute_cluster_score(self) -> float:
        return -1