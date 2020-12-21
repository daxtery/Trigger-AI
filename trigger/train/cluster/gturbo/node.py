import time
import numpy as np

from typing import List, Dict

class Node:

    def __init__(self, protype, error: float, id: int, error_cycle: int, radius: float) -> None:

        self.protype = protype
        self.error = error
        self.topological_neighbors: Dict[int, "Node"] = {}
        self.instances = []
        self.error_cycle = error_cycle
        self.id = id
        self.radius = radius

    def __lt__(self, other: "Node"):

        return self.error > other.error

    def add_neighbor(self, neighbor: "Node") -> None:

        self.topological_neighbors[neighbor.id] = neighbor

    def add_instance(self, instance) -> None:

        self.instances.append(instance)

    def remove_neighbor(self, neighbor: "Node") -> None:

        self.topological_neighbors.pop(neighbor.id)

    def remove_instance(self, instance):

        self.instances.remove(instance)

    def update_error_cycle(self, cycle: int) -> None:

        self.error_cycle = cycle


