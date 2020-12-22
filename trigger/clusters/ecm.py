from trigger.clusters.processor import Processor
from typing import Any, Dict, List, Optional, Tuple

import numpy

from scipy.spatial.distance import cdist

import numpy as np

from enum import Enum


class Cluster:
    def __init__(self, tag: str, center: numpy.ndarray, index: int) -> None:
        self.center = center
        self.radius = 0
        self.tags = [tag]
        self.index = index

    def add_radius(self, tag: str, instance: numpy.ndarray) -> None:
        self.tags.append(tag)

    def _adapt(self, distance: float, instance: numpy.ndarray):
        direction = instance - self.center
        self.radius = distance / 2
        self.center = instance - (direction / np.linalg.norm(direction)) * self.radius

    def add_threshold(self, distance: float, tag: str, instance: numpy.ndarray) -> None:
        self.add_radius(tag, instance)
        self._adapt(distance, instance)

    def update_radius(self, tag: str, instance: numpy.ndarray) -> None:
        pass

    def update_threshold(self, distance: float, tag: str, instance: numpy.ndarray) -> None:
        self.update_radius(tag, instance)
        self._adapt(distance, instance)

    def remove(self, tag: str) -> None:
        self.tags.remove(tag)


class SearchResultType(Enum):
    RADIUS = 1
    THRESHOLD = 2
    OUTSIDE = 3


class ECM(Processor):

    def __init__(self, distance_threshold: float) -> None:
        self.clusters: Dict[int, Cluster] = {}
        self.distance_threshold = distance_threshold
        self.tag_to_cluster: Dict[str, int] = {}
        self.cluster_index = 0

        self.cached_cluster_keys: List[int] = []
        self.cached_cluster_centers: List[float] = []
        self.cached_cluster_radiuses: List[float] = []

    def update(self, tag: str, instance: numpy.ndarray) -> None:
        result, (searched_index, searched_distance) = self._search_index_and_distance(instance)
        old_index = self.get_cluster_by_tag(tag)
        old_cluster = self.clusters[old_index]

        if result == SearchResultType.OUTSIDE:
            self._remove_from_cluster(old_cluster, tag)

            cluster = self._create_cluster(tag, instance)
            index = cluster.index

        elif result == SearchResultType.RADIUS:
            if searched_index == old_index:
                old_cluster.update_radius(tag, instance)

                index = searched_index
            else:
                self._remove_from_cluster(old_cluster, tag)

                new_cluster = self.clusters[searched_index]
                new_cluster.add_radius(tag, instance)

                index = searched_index

        # elif result == SearchResultType.THRESHOLD:
        else:
            if searched_index == old_index:
                old_cluster.update_threshold(searched_distance, tag, instance)

                index = searched_index
            else:
                self._remove_from_cluster(old_cluster, tag)

                new_cluster = self.clusters[searched_index]
                new_cluster.add_threshold(searched_distance, tag, instance)

                index = searched_index

        self.tag_to_cluster[tag] = index
        self._invalidate_cached()

    def _remove_from_cluster(self, cluster: Cluster, tag: str) -> None:
        cluster.remove(tag)
        if len(cluster.tags) == 0:
            del self.clusters[cluster.index]

    def _create_cluster(self, tag: str, instance: numpy.ndarray) -> Cluster:
        cluster = Cluster(tag, instance, self.cluster_index)
        self.clusters[self.cluster_index] = cluster
        self.cluster_index += 1
        return cluster

    def remove(self, tag: str) -> None:
        index = self.get_cluster_by_tag(tag)
        cluster = self.clusters[index]

        del self.tag_to_cluster[tag]

        self._remove_from_cluster(cluster, tag)
        self._invalidate_cached()

    def get_cluster_by_tag(self, tag: str) -> int:
        return self.tag_to_cluster[tag]

    def get_tags_in_cluster(self, cluster_id: int) -> List[str]:
        return self.clusters[cluster_id].tags

    def get_cluster_ids(self) -> List[int]:
        return list(self.clusters.keys())

    def process(self, tag: str, instance: numpy.ndarray) -> None:
        if len(self.clusters) == 0:
            cluster = self._create_cluster(tag, instance)

        else:
            search_result, (index, distance) = self._search_index_and_distance(instance)

            if search_result == SearchResultType.RADIUS:
                cluster = self.clusters[index]
                cluster.add_radius(tag, instance)

            elif search_result == SearchResultType.THRESHOLD:
                cluster = self.clusters[index]
                cluster.add_threshold(distance, tag, instance)

            # search_result == SearchResultType.OUTSIDE
            else:
                cluster = self._create_cluster(tag, instance)

        self.tag_to_cluster[tag] = cluster.index
        self._invalidate_cached()

    def _invalidate_cached(self):
        self.cached_cluster_keys = []
        self.cached_cluster_centers = []
        self.cached_cluster_radiuses = []

    def _ensure_cached(self):
        if not self.cached_cluster_keys and len(self.clusters) > 0:
            self.cached_cluster_keys = []
            self.cached_cluster_centers = []
            self.cached_cluster_radiuses = []
            for index, cluster in self.clusters.items():
                self.cached_cluster_keys.append(index)
                self.cached_cluster_centers.append(cluster.center)
                self.cached_cluster_radiuses.append(cluster.radius)


    def _search_index_and_distance(self, instance: Any) -> \
            Tuple[SearchResultType, Tuple[int, int]]:

        self._ensure_cached()

        distances = cdist(
            np.array([instance]),
            np.array(self.cached_cluster_centers),
            'euclidean'
        )[0]

        diffs = distances - self.cached_cluster_radiuses

        possible_indexes = np.where(diffs <= 0)[0]

        possible = distances[possible_indexes]

        min_index = None if possible.size == 0 else possible_indexes[possible.argmin()]

        if min_index is not None:
            return SearchResultType.RADIUS, (self.cached_cluster_keys[min_index], distances[min_index])

        distances_plus_radiuses = np.add(distances, self.cached_cluster_radiuses)
        lowest_distance_and_radius_index = np.argmin(distances_plus_radiuses)
        lowest_distance_and_radius = distances_plus_radiuses[lowest_distance_and_radius_index]

        actual_index = self.cached_cluster_keys[lowest_distance_and_radius_index]

        if lowest_distance_and_radius > 2 * self.distance_threshold:
            return SearchResultType.OUTSIDE, (actual_index, lowest_distance_and_radius)

        else:
            return SearchResultType.THRESHOLD, (actual_index, lowest_distance_and_radius)

    def describe(self) -> Dict[str, Any]:
        """
        This describes this clustering algorithm's parameters
        """

        return {
            "name": "ECM",
            "parameters": {
                "distance threshold": self.distance_threshold
            }
        }

    def safe_file_name(self) -> str:
        return f"ECM = distance_threshold={self.distance_threshold}"

    def predict(self, instance: Any) -> int:
        search_result, (index, _) = self._search_index_and_distance(instance)

        # FIXME: What should predict do in this case?
        if search_result == SearchResultType.OUTSIDE:
            return index

        elif search_result == SearchResultType.THRESHOLD:
            return index

        #elif search_result == SearchResultType.RADIUS:
        else:
            return index